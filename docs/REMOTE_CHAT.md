# Remote browser chat

Use Camelid from another device while inference stays on the host computer. For local setup, start with the [quick start](../README.md#quick-start).

## Trusted private LAN

> [!WARNING]
> A non-loopback listener requires authentication and either TLS or an explicit cleartext
> acknowledgement. For browser Chat on a trusted private LAN, bind
> the laptop's specific LAN address, start with a model, and use the restricted surface:
>
> ```bash
> camelid serve --addr <LAPTOP-LAN-IP>:8181 --model models/Llama-3.2-3B-Instruct-Q8_0.gguf --api-key-file ./camelid-api.key --lan-chat-only --allow-cleartext-remote
> ```
>
> The phone is only a web interface; inference stays on the laptop. Plain HTTP is not encrypted, and
> `--allow-cleartext-remote` acknowledges that risk rather than protecting the traffic. Chat history
> is still stored separately in each browser. Run `camelid lan-key` on the laptop to
> create or display the key shared with the phone. The mobile Chat model selector may switch only
> among local GGUF files already in the configured models directory. See
> [configuration](CONFIGURATION.md) for the exact route boundary, key rotation, firewall
> guidance, CORS, TLS, and remote-deployment options.

## Across networks with Tailscale

For private Chat across different networks, keep Camelid on `127.0.0.1`, install Tailscale on both
devices, and run `camelid remote-chat start` after the authenticated `--lan-chat-only` listener is
healthy. Camelid prints a tailnet-only HTTPS URL; the browser still requires the Camelid API key.
This workflow never enables Tailscale Funnel. See [configuration](CONFIGURATION.md#private-cross-network-browser-chat-with-tailscale).

