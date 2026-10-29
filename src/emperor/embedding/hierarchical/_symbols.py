"""HAT v2 wire alphabet. FE/FF never occur in valid UTF-8 text.

The paper leaves the numeric control IDs unspecified. This mapping is protocol
version 2; the previous Emperor adaptation used additional IDs 256 and 257.
"""

BYTE_VALUES = 256
OUTPUT_SYMBOLS = 256
END_OF_TOKEN = 0xFE
END_OF_DOCUMENT = 0xFF
HAT_PROTOCOL_VERSION = 2
