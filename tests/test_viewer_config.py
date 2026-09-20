"""Shadow toggle in the viewer config.

Viser hardcodes PCSS shadows and its default light is a 3-cascade 1024^2 shadow
map recomputed every frame, so with shadows on each frame draws the scene ~4x:
one colour pass plus three shadow depth passes. The workstation URDFs are
1.4-1.9M triangles per pass, which a discrete GPU absorbs and an integrated one
does not — the reported "replay is laggy on my PC" with the same episode playing
smoothly elsewhere. Replay keeps shadows on to match viser's own default and
leave the picture unchanged; `shadows: false` is the escape hatch for a client
that cannot afford them.

`app.run` passes this straight to `scene.configure_default_lights(cast_shadow=)`;
these cover the config contract it reads.
"""

from linker_sim_viser.config import ViewerConfig, load_viewer_config


def test_shadows_default_on():
    """Default matches viser, so enabling the knob changes no visuals."""
    assert ViewerConfig().shadows is True


def test_absent_key_defaults_on(tmp_path):
    """A config predating the knob renders exactly as it did before."""
    cfg = tmp_path / "viewer.yaml"
    cfg.write_text("port: 8080\n")

    assert load_viewer_config(cfg).shadows is True


def test_shadows_can_be_disabled(tmp_path):
    """The point of the knob: opting out on a GPU that cannot keep up."""
    cfg = tmp_path / "viewer.yaml"
    cfg.write_text("shadows: false\n")

    assert load_viewer_config(cfg).shadows is False


def test_shipped_config_keeps_shadows_on():
    """Pins the shipped default: this is what a customer gets out of the box."""
    assert load_viewer_config("configs/viewer.yaml").shadows is True


def test_shipped_config_caps_the_trail():
    """The cap is only meaningful if the shipped config actually sets it."""
    assert load_viewer_config("configs/viewer.yaml").trails.max_points == 500
