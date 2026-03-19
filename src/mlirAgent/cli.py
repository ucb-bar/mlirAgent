import argparse
import json

from mlirAgent.tools.build import run_build
from mlirAgent.tools.compiler import run_compile
from mlirAgent.tools.provenance import MLIRProvenanceTracer
from mlirAgent.tools.trace_provenance import trace_provenance
from mlirAgent.tools.verifier import verify_output


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _print_result(result: dict) -> None:
    print(json.dumps(result, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    result = run_build(
        target=args.target,
        fast_mode=args.fast,
        clean=args.clean,
        reconfigure=args.reconfigure,
    )
    _print_result(result)


def cmd_compile(args: argparse.Namespace) -> None:
    mlir_text = _read_text(args.input)
    result = run_compile(mlir_text, flags=args.flags)
    _print_result(result)


def cmd_verify(args: argparse.Namespace) -> None:
    ir_text = _read_text(args.ir)
    check_text = _read_text(args.check)
    result = verify_output(ir_text, check_text)
    _print_result(result)


def cmd_provenance(args: argparse.Namespace) -> None:
    tracer = MLIRProvenanceTracer()
    result = tracer.trace(args.root, args.file, args.line)
    _print_result(result)


def cmd_trace_provenance(args: argparse.Namespace) -> None:
    result_json = trace_provenance(args.root, args.file, args.line)
    print(result_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mlirAgent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Run Ninja/CMake build")
    p_build.add_argument("--target", default="install", help="Ninja target(s)")
    p_build.add_argument("--fast", action="store_true", help="Fast mode targets")
    p_build.add_argument("--clean", action="store_true", help="Run ninja clean")
    p_build.add_argument("--reconfigure", action="store_true", help="Run CMake")
    p_build.set_defaults(func=cmd_build)

    p_compile = sub.add_parser("compile", help="Run iree-compile on an MLIR file")
    p_compile.add_argument("--input", required=True, help="Path to .mlir input")
    p_compile.add_argument("--flags", nargs="*", default=[], help="Extra iree-compile flags")
    p_compile.set_defaults(func=cmd_compile)

    p_verify = sub.add_parser("verify", help="Run FileCheck on IR output")
    p_verify.add_argument("--ir", required=True, help="Path to IR file")
    p_verify.add_argument("--check", required=True, help="Path to FileCheck patterns")
    p_verify.set_defaults(func=cmd_verify)

    p_prov = sub.add_parser("provenance", help="Trace MLIR op provenance with bindings")
    p_prov.add_argument("--root", required=True, help="Artifacts root or ir_pass_history")
    p_prov.add_argument("--file", required=True, help="Source filename (e.g. input.mlir)")
    p_prov.add_argument("--line", required=True, type=int, help="Source line number")
    p_prov.set_defaults(func=cmd_provenance)

    p_trace = sub.add_parser("trace-provenance", help="Text-based provenance tracing")
    p_trace.add_argument("--root", required=True, help="Artifacts root")
    p_trace.add_argument("--file", required=True, help="Source filename")
    p_trace.add_argument("--line", required=True, type=int, help="Source line number")
    p_trace.set_defaults(func=cmd_trace_provenance)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
