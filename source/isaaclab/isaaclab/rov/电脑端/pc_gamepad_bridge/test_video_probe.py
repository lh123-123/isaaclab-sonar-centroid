import struct
import unittest

from video_probe import rtp_packet


def packet(payload, flags=0x80):
    return struct.pack('!BBHII', flags, 0xe0, 65535, 1234, 7) + payload


class VideoProbeTests(unittest.TestCase):
    def test_single_and_fragmented_h264(self):
        self.assertEqual(rtp_packet(packet(b'\x67\xaa')), (65535, 1234, 7, True, {7}))
        self.assertEqual(rtp_packet(packet(b'\x7c\x85\xaa'))[-1], {5})
        self.assertEqual(rtp_packet(packet(b'\x7c\x05\xaa'))[-1], set())

    def test_aggregation_validates_every_length(self):
        self.assertEqual(rtp_packet(packet(b'\x78\x00\x02\x67\xaa\x00\x01\x68'))[-1], {7, 8})
        for payload in (b'\x78', b'\x78\x00\x03\x67', b'\x78\x00\x01\x67\xaa'):
            self.assertIsNone(rtp_packet(packet(payload)))

    def test_padding_and_extension_bounds(self):
        self.assertEqual(rtp_packet(packet(b'\x65\xaa\x02', 0xa0))[-1], {5})
        self.assertIsNone(rtp_packet(packet(b'\x65\x00', 0xa0)))
        self.assertIsNone(rtp_packet(packet(b'\x65\xff', 0xa0)))
        self.assertIsNone(rtp_packet(packet(b'\x00\x00\x01\x00\xaa', 0x90)))
