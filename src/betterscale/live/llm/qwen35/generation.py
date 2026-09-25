"""Synchronous greedy MTP2 protocol over a root-owned resident continuation.

Speculative writes are not commits. GDN selection, shifted draft validity and
host prefix identity advance together after verification; no hidden history is
kept beyond the current short wave and one boundary vector per resident.
"""

import torch


def greedy(logits_or_ids):
    return logits_or_ids if logits_or_ids.ndim == 1 else logits_or_ids.argmax(-1)


def generate(root, prompt, max_new_tokens, *, speculative=True, eos_token_ids=()):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    table = root.residents_table
    lease, cached = table.acquire(list(prompt))
    seat = lease.seat
    c = root.continuation
    emitted = []
    proposed = accepted_drafts = 0
    try:
        for position in range(cached, len(prompt)):
            token = prompt[position]
            slots = table.reserve(lease, position + 1)
            if position:
                root.draft_step(
                    [token],
                    c.anchor_hidden.tensor[seat : seat + 1],
                    position=position - 1,
                    slots=slots[:position],
                )
            hidden, logits = root.target_step(
                [token],
                seat=seat,
                position=position,
                slots=slots,
                accepted=int(c.selection.tensor[seat].item()),
            )
            c.anchor_hidden.tensor[seat].copy_(hidden[-1])
            c.anchor_token.tensor[seat] = greedy(logits)[-1]
            c.selection.tensor[seat] = 1
            c.target_cursor.tensor[seat] = position + 1
            c.draft_cursor.tensor[seat] = position
            table.commit(lease, [token])
        while len(emitted) < max_new_tokens:
            position = len(table.seats[seat].tokens)
            anchor = int(c.anchor_token.tensor[seat].item())
            width = 3 if speculative else 1
            slots = table.reserve(lease, position + width)
            seed, logits = root.draft_step(
                [anchor],
                c.anchor_hidden.tensor[seat : seat + 1],
                position=position - 1,
                slots=slots[:position],
            )
            tokens = [anchor]
            if speculative:
                first = int(greedy(logits)[-1].item())
                _, logits = root.draft_step(
                    [first], seed, position=position, slots=slots[: position + 1]
                )
                tokens.extend([first, int(greedy(logits)[-1].item())])
                proposed += 2
                c.proposal.tensor[seat].copy_(
                    torch.tensor(tokens[1:], device=root.live_device)
                )
            hidden, logits = root.target_step(
                tokens,
                seat=seat,
                position=position,
                slots=slots,
                accepted=int(c.selection.tensor[seat].item()),
            )
            predictions = greedy(logits).tolist()
            count = 1
            while count < len(tokens) and tokens[count] == predictions[count - 1]:
                count += 1
            count = min(count, max_new_tokens - len(emitted))
            for i in range(count):
                if tokens[i] in eos_token_ids:
                    count = i + 1
                    break
            accepted_drafts += count - 1
            # Replace recursive draft hidden seeds with verified target seeds.
            # The preceding first-pass row (position-1) was already committed.
            if count > 1:
                root.draft_step(
                    tokens[1:count],
                    hidden[: count - 1],
                    position=position,
                    slots=slots[: position + count - 1],
                )
            c.anchor_hidden.tensor[seat].copy_(hidden[count - 1])
            c.anchor_token.tensor[seat] = predictions[count - 1]
            c.selection.tensor[seat] = count
            c.target_cursor.tensor[seat] = position + count
            c.draft_cursor.tensor[seat] = position + count - 1
            table.commit(lease, tokens[:count])
            emitted.extend(tokens[:count])
            if emitted[-1] in eos_token_ids:
                break
        torch.npu.synchronize(root.live_device)
        return {
            "token_ids": emitted,
            "seat": seat,
            "cached_tokens": cached,
            "proposed": proposed,
            "accepted_drafts": accepted_drafts,
        }
    except BaseException:
        # Do not publish a hot identity after any partially completed writer.
        # Drain our synchronous device work before invalidating/clearing it.
        torch.npu.synchronize(root.live_device)
        table.finish(lease)
        table.evict(seat)
        raise
    finally:
        if table.seats[seat].request == lease.request:
            table.finish(lease)
