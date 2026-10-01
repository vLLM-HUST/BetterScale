import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
import pytest


def helpers():
    tree=ast.parse(Path(__file__).with_name('numerical_entry.py').read_text())
    def idle(core):
        if core.busy:raise RuntimeError('Not retired')
        return core.scheduler
    ns=dict(Future=Future,idle=idle)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                                 and n.name in ('snapshot','observe','service')],type_ignores=[]),
                 'numerical_entry.py','exec'),ns)
    seat=NS(index=0,epoch=5,cursor=296,cache_salt='case',tokens=(1,2),fence=16,
            blocks=NS(blocks=((NS(block_id=1),),)))
    return ns,NS(busy=False,scheduler=NS(residents=NS(seats=[seat]),processed_step_seq=16))


def test_snapshot_reports_native_identity_without_mutation():
    h,c=helpers();r=h['observe'](c,'case')
    assert r==dict(seat=0,epoch=5,cursor=296,pages=[[1]],retired_step=16,processed_step=16)
    assert c.scheduler.residents.seats[0].epoch==5


def test_pending_observation_waits_for_actual_retirement():
    h,c=helpers();c.busy=True;f=h['observe'](c,'case')
    h['service'](c);assert not f.done()
    c.busy=False;h['service'](c);assert f.result()['pages']==[[1]]
    assert c._numerical_pending is None


def test_duplicate_pending_is_rejected():
    h,c=helpers();c.busy=True;h['observe'](c,'case')
    with pytest.raises(RuntimeError,match='pending'):h['observe'](c,'case')


def test_failed_lookup_is_not_a_successful_observation():
    h,c=helpers();c.busy=True;f=h['observe'](c,'missing')
    c.busy=False;h['service'](c)
    with pytest.raises(ValueError,match='resident'):f.result()
