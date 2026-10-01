"""Pinned native AsyncLLM pools; owner-specific utilities never broadcast.

This is a local experiment actor, not an HTTP service or a second DP coordinator.
Native FIRST_REQ wakes EP peers. The parent controls quiescent session handoffs.
"""
import asyncio
import os
import traceback
import uuid

METHODS = frozenset(('pd_export_retired', 'pd_wait_export', 'pd_finish_export',
                     'pd_drop_target', 'pd_import_target', 'pd_probe_retired'))

async def owner_utility(client, owner, method, *args):
    if type(owner) is not int or not 0 <= owner < len(client.core_engines):
        raise ValueError('Invalid attention owner')
    if method not in METHODS:
        raise ValueError('Unqualified owner utility')
    # Pinned DPLB public call_utility_async broadcasts and returns only rank0.
    return await client._call_utility_async(
        method, *args, engine=client.core_engines[owner])

async def run(role, connection, output):
    from pd_model_probe import prepare_worker, engine_options
    is_p = prepare_worker(role, native_async=True)
    options = engine_options(is_p)
    if os.environ.get('BETTERSCALE_NUMERICAL_RESIDENCY'):
        if is_p:raise ValueError('Numerical probe supports D only')
        import numerical_entry
        options.update(worker_cls='numerical_entry.Worker',scheduler_cls='numerical_entry.Scheduler')
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import DPLBAsyncMPClient
    count = 1 if is_p else 3
    options.update(data_parallel_size=count, data_parallel_size_local=count,
                   data_parallel_address='127.0.0.1',
                   data_parallel_rpc_port=29653 if is_p else 29673,
                   disable_log_stats=True)
    model = None
    try:
        model = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        client = model.engine_core
        assert len(client.core_engines) == count
        if not is_p:
            assert isinstance(client, DPLBAsyncMPClient)
        connection.send(('ready', role))
        while True:
            # Keep native output/coordinator tasks live while awaiting commands.
            op, args = await asyncio.to_thread(connection.recv)
            if op == 'stop':
                break
            owner = args['owner']
            if type(owner) is not int or not 0 <= owner < count:
                raise ValueError('Invalid request owner')
            async def utility(method, *values):
                return await owner_utility(client, owner, method, *values)
            if op == 'generate':
                tokens, salt = args['tokens'], args['salt']
                prompt = (dict(prompt=tokens, cache_salt=salt) if isinstance(tokens, str)
                          else dict(prompt_token_ids=tokens, cache_salt=salt))
                params = SamplingParams(temperature=0, max_tokens=args['n'],
                    ignore_eos=True, logprobs=5 if args.get('diagnostic') else None,
                    output_kind=RequestOutputKind.FINAL_ONLY)
                final = None
                async for value in model.generate(prompt, params, uuid.uuid4().hex,
                                                   data_parallel_rank=owner):
                    final = value
                assert final is not None and final.finished
                seq = final.outputs[0]
                result = dict(prompt_token_ids=final.prompt_token_ids,
                    token_ids=list(seq.token_ids), text=seq.text,
                    cached=final.num_cached_tokens,
                    logprobs=[{str(k):v.logprob for k,v in row.items()}
                              for row in seq.logprobs] if seq.logprobs is not None else None)
            elif op == 'export':
                result = await utility('pd_export_retired', args['tokens'], args['salt'],
                                       args.get('dense_start', 0), args.get('stream_store'))
            elif op == 'wait-export':
                result = await utility('pd_wait_export', args['transfer_id'])
            elif op == 'finish-export':
                result = await utility('pd_finish_export', args['transfer_id'])
            elif op == 'drop':
                result = await utility('pd_drop_target', args['salt'])
            elif op == 'import':
                result = await utility('pd_import_target', args['payload'], args['salt'])
            elif op == 'observe':
                result = await utility('pd_probe_retired', args['salt'])
            elif op == 'append':
                result = args['tokens'] + model.get_tokenizer().encode(
                    args['text'], add_special_tokens=False)
            else:
                raise ValueError(op)  # In particular, no manual wake operation.
            connection.send(('ok', result))
    except BaseException:
        error = traceback.format_exc()
        (output / (role + '-failure.txt')).write_text(error)
        try:
            connection.send(('error', error))
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        if model is not None:
            model.shutdown()
        connection.close()

def worker(role, connection, output):
    asyncio.run(run(role, connection, output))
