import asyncio
import fcntl
import pytest
from retire_online_cache import retire
from session import Directory


URLS=("http://10.244.1.16:55581","http://10.244.2.32:55586")


def test_retirement_never_creates_a_missing_frontend_directory(tmp_path):
    with pytest.raises(ValueError,match="existing frontend"):
        asyncio.run(retire(tmp_path,tmp_path/"receipt.json",*URLS))
    assert not list(tmp_path.iterdir())


def test_retirement_refuses_live_controller_lock(tmp_path):
    path=tmp_path/"sessions.sqlite";Directory(path)
    with open(str(path)+".controller.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            asyncio.run(retire(tmp_path,tmp_path/"receipt.json",*URLS))
    assert not (tmp_path/"receipt.json").exists()


def test_retirement_refuses_unfinished_numerical_owner(tmp_path):
    d=Directory(tmp_path/"sessions.sqlite");d.create("s","identity","P")
    d.claim("s","P","identity")
    with pytest.raises(ValueError,match="unfinished numerical"):
        asyncio.run(retire(tmp_path,tmp_path/"receipt.json",*URLS))
