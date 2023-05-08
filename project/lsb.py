from PIL import Image
from math import ceil, floor
import numpy as np
from range_table import RangeTables
from bitarray import bitarray

class LSB:
    
    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)

        self.channels_capacities = None # Capacity (n_bits) for every channel
        self.k = None # number of embedded bits in LSB
        self.stego_result = None # Resulting pixels of the stego image (list of lists of pixels)
        self.flattened_stego = None # Resulting pixels of the stego image (list of color tuples)

    def calculate_ranges(self, k):
        self.k = k
        self.channels_capacities = [k * len(channel) for channel in self.channels_px ]

    def get_capacity(self):
        if self.channels_capacities is None:
            raise RuntimeError('calculate_ranges must be called before capacity can be calculated!')
        sum = 0
        for channel in range(self.channels_count):
            sum += self.channels_capacities[channel]
        return sum, sum # theoretical, actual 
    
    def print_capacity(self):
        try:
            _, capacity = self.get_capacity()
            print(f"Calculated capacity: \n ={capacity} b\n ~{capacity // (1024 * 8)} KB")
        except RuntimeError as e:
            print(e)

    def get_bit_to_embed(self, data, data_ptr, result_ptr, capacity):
        # returns (bit_to_embed, data_ptr_delta)
        if result_ptr == capacity and data_ptr == result_ptr:
            return 1, 0 # full file, end with a 1
        elif data_ptr == result_ptr:
            if data_ptr == len(data):
                return 1, 0 # data ended, cap it off with a one
            else: 
                return data[data_ptr], 1 # n-th bit of result is n-th bit of data, there is still data to read
        else:
            return 0, 0 # data ended and capped off - pad with zeros to the end

    def calculate_stego(self, data: bytearray): 
        self.stego_result = []
        self.flattened_stego = []
        capacity, _ = self.get_capacity()
        mask = (0b11111111 >> self.k) << self.k
        data_ptr = 0
        result_ptr = 0
        for channel in range(self.channels_count):
            result = []
            for px in self.channels_px[channel]:
                embedded_num = 0
                for j in range(self.k):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    embedded_num += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1

                px &= mask
                px |= embedded_num
                result.append(px)
            self.stego_result.append(result)

        self.flattened_stego = []
        for i in range(len(self.stego_result[0])):
            color = tuple(self.stego_result[j][i] for j in range(len(self.stego_result)))
            self.flattened_stego.append(color)

        data = data[data_ptr:]
        return data

    def get_result(self, out_name):
        im = Image.new(self.im.mode, self.im.size)
        im.putdata(self.flattened_stego)
        im.save(out_name)
        return im

    def hide_data(self, data: bytearray, k, out_name: str):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(k)
        new_data = self.calculate_stego(data)
        result = self.get_result(out_name)
        return result, new_data


class ReverseLSB:

    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)

        self.channels_capacities = None # Capacity (n_bits) for every channel
        self.k = None # Number of embedded bits in LSB
        self.result = None # Resulting pixels of the stego image

    def calculate_ranges(self, k):
        self.k = k
        self.channels_capacities = [k * len(channel) for channel in self.channels_px ]

    def reverse_stego(self):
        self.result = bitarray()
        for channel in range(self.channels_count):
            for px in self.channels_px[channel]:
                for _ in range(self.k):
                    self.result.append(px % 2)
                    px >>= 1

        # remove padding
        initial_len = self.result
        while len(self.result) > 0 and self.result.pop() == 0:
            pass
        if len(self.result) == 0:
            # the message was all 0s
            self.result = bitarray([0] * initial_len)

        return self.result
    
    def extract(self, k):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(k)
        result = self.reverse_stego()
        return result
