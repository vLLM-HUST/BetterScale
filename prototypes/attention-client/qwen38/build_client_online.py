"""Append the route-generation consumer; producer opt-in is a separate ABI bit."""

from pathlib import Path


def append_online(client):
    assert "void\nneural_collect_online(" not in client
    return client + "\n" + Path(__file__).with_name("client_online.cpp").read_text()
