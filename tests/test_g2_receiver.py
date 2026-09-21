"""RPC failure policy: local shutdown is expected, remote errors reach actor."""
from queue import Queue
import threading
import grpc
import pytest
from g2_local import runtime


class FailedStream:
    def __iter__(self):
        raise grpc.RpcError('connection lost')


@pytest.mark.parametrize('closing', [True, False])
def test_receive_error_is_reported_unless_shutting_down(closing):
    stop, output = threading.Event(), Queue()
    if closing:
        stop.set()
    runtime.receive_parameters(FailedStream(), output, stop)
    if closing:
        assert output.empty()
    else:
        assert isinstance(output.get_nowait(), grpc.RpcError)
