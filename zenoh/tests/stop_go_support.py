"""A minimal executor for node-protocol tests: always stopped, steps finish at once."""

from __future__ import annotations

from threading import Lock

from config import ExecutorState, JSON_ENCODING, NodeState, PlanarPose


def executor_feedback(state: NodeState, motion="stopped", inference_id=None, session="test-executor",
                      pose: PlanarPose | None = None) -> ExecutorState:
    return ExecutorState(executor_session_id=session, node_session_id=state.node_session_id,
                         control_epoch=state.control_epoch, cycle_id=state.cycle.cycle_id, motion=motion,
                         inference_id=inference_id, pose=pose or PlanarPose())


def completed_feedback(state: NodeState) -> ExecutorState:
    """Stopped feedback that also completes an authorized step."""
    return executor_feedback(state, inference_id=state.cycle.inference_id if state.cycle.phase == "moving" else None)


def report_stopped(transport) -> None:
    """Feed one stopped/completed report into a node TransportState."""
    transport.receive_executor(completed_feedback(transport.snapshot()))


class EchoExecutor:
    """Over Zenoh: answers every node state with stopped/completed feedback."""

    def __init__(self, session, keys):
        self.lock, self.closed = Lock(), False
        self.publisher = session.declare_publisher(keys["executor_state"], encoding=JSON_ENCODING)
        self.subscriber = session.declare_subscriber(keys["state"], self._on_state)

    def _on_state(self, sample):
        state = NodeState.model_validate_json(sample.payload.to_bytes())
        with self.lock:  # Never put while close() undeclares the publisher.
            if not self.closed and state.stage == "inference" and state.node_session_id:
                self.publisher.put(completed_feedback(state).model_dump_json())

    def close(self):
        with self.lock:
            self.closed = True
        self.subscriber.undeclare()
        with self.lock:
            self.publisher.undeclare()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
