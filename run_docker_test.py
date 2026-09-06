import subprocess


def run_docker_smoke_test():
    print("Building docker image...")
    subprocess.run(
        ["docker", "build", "-t", "marl-supply-chain-phase0", "."], check=True
    )

    print("Running deterministic fixture locally...")
    out_local = subprocess.check_output(
        [".\\.venv\\Scripts\\python.exe", "smoke_test.py"], text=True
    ).strip()

    print("Running deterministic fixture in docker...")
    out_docker = subprocess.check_output(
        ["docker", "run", "--rm", "marl-supply-chain-phase0"], text=True
    ).strip()

    with open("host_hash.txt", "w") as f:
        f.write(out_local)

    with open("container_hash.txt", "w") as f:
        f.write(out_docker)

    print(f"Local hash: {out_local}")
    print(f"Docker hash: {out_docker}")

    if out_local == out_docker:
        print("Hash equality: PASS")
    else:
        print("Hash equality: FAIL")


if __name__ == "__main__":
    run_docker_smoke_test()
