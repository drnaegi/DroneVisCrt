import functools
import json
import socket
import subprocess


@functools.lru_cache(maxsize=128)
def resolve_ipv4_all(hostname):
    result = subprocess.run(
        [
            "curl", "-fsS", "--connect-timeout", "5", "--max-time", "15",
            "--resolve", "dns.alidns.com:443:223.5.5.5",
            "-H", "accept: application/dns-json",
            "https://dns.alidns.com/resolve?name=" + hostname + "&type=A",
        ],
        capture_output=True, text=True, check=True,
    )
    answers = json.loads(result.stdout).get("Answer", [])
    addresses = [answer["data"] for answer in answers if answer["type"] == 1]
    if not addresses:
        raise OSError("No IPv4 DNS answer for " + hostname)
    return addresses


def resolve_ipv4(hostname):
    return resolve_ipv4_all(hostname)[0]


def enable():
    original = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        hostname = host.decode() if isinstance(host, bytes) else host
        if hostname and "." in hostname:
            try:
                socket.inet_aton(hostname)
            except OSError:
                addresses = resolve_ipv4_all(hostname)
                return [result for address in addresses for result in original(address, port, socket.AF_INET, type, proto, flags)]
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo
