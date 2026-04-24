import argparse
import json

import zmq


def _send_request(socket, payload):
    socket.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    response_raw = socket.recv()
    return json.loads(response_raw.decode("utf-8"))


def _print_response(response):
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _interactive_loop(socket):
    print('Interactive mode. Commands: status | set_anchor | search <query> | stop | quit')
    while True:
        try:
            line = input("robot> ").strip()
        except EOFError:
            print()
            break

        if not line:
            continue
        if line in {"quit", "exit"}:
            break
        if line == "status":
            _print_response(_send_request(socket, {"command": "status"}))
            continue
        if line == "set_anchor":
            _print_response(_send_request(socket, {"command": "set_anchor"}))
            continue
        if line == "stop":
            _print_response(_send_request(socket, {"command": "stop"}))
            break
        if line.startswith("search "):
            query = line[len("search ") :].strip()
            _print_response(_send_request(socket, {"command": "search", "query": query}))
            continue

        print("Unsupported command.")


def _build_argparser():
    parser = argparse.ArgumentParser(description="Socket client for real_main_on_robot.py")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="real_main_on_robot 服务地址")
    parser.add_argument("--port", type=int, default=6000, help="real_main_on_robot 服务端口")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="发送/接收超时（毫秒）")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="查询当前状态")
    subparsers.add_parser("set_anchor", help="用当前帧手动设置 anchor")

    search_parser = subparsers.add_parser("search", help="发送搜索请求")
    search_parser.add_argument("query", type=str, help="搜索指令")

    subparsers.add_parser("stop", help="停止 real_main_on_robot 服务")
    subparsers.add_parser("interactive", help="进入交互模式")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(args.timeout_ms))
    socket.connect(f"tcp://{args.host}:{args.port}")

    try:
        if args.command == "status":
            response = _send_request(socket, {"command": "status"})
            _print_response(response)
        elif args.command == "set_anchor":
            response = _send_request(socket, {"command": "set_anchor"})
            _print_response(response)
        elif args.command == "search":
            response = _send_request(socket, {"command": "search", "query": args.query})
            _print_response(response)
        elif args.command == "stop":
            response = _send_request(socket, {"command": "stop"})
            _print_response(response)
        elif args.command == "interactive":
            _interactive_loop(socket)
        else:
            raise SystemExit("Please choose a command: status | set_anchor | search | stop | interactive")
    except zmq.error.Again:
        raise SystemExit("Socket timeout: no response from real_main_on_robot service")
    finally:
        socket.close(linger=0)
        context.term()
