import json
import pytest
from test_g2_training_config import valid_payload


def test_offline_accepts_shared_motion_profile_only_explicitly_without_permission(tmp_path):
    from g2_local.offline_pretrain import load_offline_config
    payload = valid_payload()
    payload['requested_motion'] = True
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_offline_config(path)
    config = load_offline_config(path, accept_motion_profile=True)
    assert config.requested_motion is True
    assert config.motion_permitted is False
