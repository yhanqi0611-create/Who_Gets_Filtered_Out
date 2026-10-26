from wscaudit.cli import SUBCOMMANDS, build_parser, load_paths


def test_parser_lists_all_subcommands() -> None:
    parser = build_parser()
    # argparse stores subparsers on a private _subparsers action
    subparser_actions = [
        a for a in parser._actions if getattr(a, "dest", None) == "command"
    ]
    assert subparser_actions, "expected a subparsers action named command"
    choices = set(subparser_actions[0].choices or [])
    assert choices == set(SUBCOMMANDS)


def test_load_paths_reads_defaults() -> None:
    cfg = load_paths()
    assert cfg["data_root"]
    assert cfg["out_root"]
