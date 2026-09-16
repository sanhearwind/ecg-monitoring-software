# Device protocol

The desktop host application was developed independently, but the serial protocol described here was **not independently designed**. It directly follows the fixed 10-byte protocol used by the course-provided STM32 baseline/template firmware. `host/protocol.py` implements a newly written host-side parser for that existing interoperability interface.

```text
byte 0      Module ID
byte 1      DataHead
byte 2      Secondary ID
bytes 3-8   Payload bytes 0-5
byte 9      Checksum
```

## Framing

- `Module ID` is below `0x80` and marks the start of a frame.
- Bytes 1-9 have bit 7 set while transmitted.
- `DataHead` stores the original high bits of the secondary ID and payload.
- The parser restarts if a new module byte appears before a frame is complete.
- A frame is returned only when all ten bytes arrive and its checksum is valid.

## Checksum

The checksum is calculated from the packed bytes 0-8. The low seven bits of the
sum must match the low seven bits of byte 9.

## ECG messages

| Module / secondary ID | Direction | Meaning |
|---|---|---|
| `0x10 / 0x02` | device to host | Big-endian 16-bit ECG waveform sample |
| `0x10 / 0x03` | device to host | Lead status (`0` connected, `1` lead off) |
| `0x10 / 0x04` | device to host | Big-endian 16-bit heart-rate value |

The protocol documents interoperability with the course prototype. It is not
claimed as an independently invented wire format.
