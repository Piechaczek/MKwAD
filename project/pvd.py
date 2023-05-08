from PIL import Image
from math import ceil, floor
import numpy as np
from range_table import RangeTables
from bitarray import bitarray

class PVD:
    
    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)
        self.blocks_count: int = 0 # Initialized after calculating blocks

        self.channels_blocks = None # Fluctuation values (differences between pixels)
        self.channels_fvs = None # Fluctuation values (differences between pixels)
        self.channels_bounds = None # (Lower bound, Upper bound) for every block
        self.channels_capacities = None # Capacity (n_bits) for every block
        self.stego_result = None # Resulting pixels of the stego image (list of lists of pixels)
        self.flattened_stego = None # Resulting pixels of the stego image (list of color tuples)
        self.calculate_blocks() # Init the lists above

    def calculate_blocks(self):
        self.channels_blocks = []
        self.channels_fvs = []
        for channel in self.channels_px:
            channel_blocks = []
            channel_fvs = []
            for i in range(1, len(channel), 2):
                channel_blocks.append((channel[i - 1], channel[i]))
                channel_fvs.append(abs(channel[i - 1] - channel[i]))
            self.channels_blocks.append(channel_blocks)
            self.channels_fvs.append(channel_fvs)
        self.blocks_count = len(self.channels_blocks[0]) # All channels have the same amount of blocks, take the first channel's blocks count

    def calculate_ranges(self, range_table):
        self.channels_bounds = []
        self.channels_capacities = []
        for channel in range(self.channels_count):
            channel_bounds = []
            channel_capacities = []
            for block in range(self.blocks_count):
                fv = self.channels_fvs[channel][block]
                l_bound, u_bound, n_bits = range_table(fv)
                channel_bounds.append((l_bound, u_bound))
                channel_capacities.append(n_bits)
            self.channels_bounds.append(channel_bounds)
            self.channels_capacities.append(channel_capacities)

    def get_capacity(self):
        if self.channels_capacities is None:
            raise RuntimeError('calculate_ranges must be called before capacity can be calculated!')
        sum = 0
        for channel in range(self.channels_count):
            for block in range(self.blocks_count):
                sum += self.channels_capacities[channel][block]
        return sum, sum - 1 # theoretical, actual 
    
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
        data_ptr = 0
        result_ptr = 0
        for channel in range(self.channels_count):
            result = []
            for block in range(self.blocks_count):
                px1, px2 = self.channels_blocks[channel][block]
                fv = self.channels_fvs[channel][block]
                n_bits = self.channels_capacities[channel][block]
                l_bound, u_bound = self.channels_bounds[channel][block]

                embedded_num = 0
                for j in range(n_bits):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    embedded_num += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1
                fv_new = embedded_num + l_bound
                diff = abs(fv - fv_new)

                if px1 >= px2 and fv_new > fv:
                    px1_new = px1 + ceil(diff / 2)
                    px2_new = px2 - floor(diff / 2)
                elif px1 < px2 and fv_new > fv:
                    px1_new = px1 - floor(diff / 2)
                    px2_new = px2 + ceil(diff / 2)
                elif px1 >= px2 and fv_new <= fv:
                    px1_new = px1 - ceil(diff / 2)
                    px2_new = px2 + floor(diff / 2)
                else:
                    px1_new = px1 + ceil(diff / 2)
                    px2_new = px2 - floor(diff / 2)
                
                result.append(px1_new)
                result.append(px2_new)
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

    def hide_data(self, data: bytearray, range_table, out_name: str):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(range_table)
        new_data = self.calculate_stego(data)
        result = self.get_result(out_name)
        return result, new_data


class ReversePVD:

    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)
        self.blocks_count: int = 0 # Initialized after calculating blocks

        self.channels_blocks = None # Fluctuation values (differences between pixels)
        self.channels_fvs = None # Fluctuation values (differences between pixels)
        self.channels_bounds = None # (Lower bound, Upper bound) for every block
        self.channels_capacities = None # Capacity (n_bits) for every block
        self.result = None # Resulting pixels of the stego image
        self.calculate_blocks() # Init the lists above

    def calculate_blocks(self):
        self.channels_blocks = []
        self.channels_fvs = []
        for channel in self.channels_px:
            channel_blocks = []
            channel_fvs = []
            for i in range(1, len(channel), 2):
                channel_blocks.append((channel[i - 1], channel[i]))
                channel_fvs.append(abs(channel[i - 1] - channel[i]))
            self.channels_blocks.append(channel_blocks)
            self.channels_fvs.append(channel_fvs)
        self.blocks_count = len(self.channels_blocks[0]) # All channels have the same amount of blocks, take the first channel's blocks count

    def calculate_ranges(self, range_table):
        self.channels_bounds = []
        self.channels_capacities = []
        for channel in range(self.channels_count):
            channel_bounds = []
            channel_capacities = []
            for block in range(self.blocks_count):
                fv = self.channels_fvs[channel][block]
                l_bound, u_bound, n_bits = range_table(fv)
                channel_bounds.append((l_bound, u_bound))
                channel_capacities.append(n_bits)
            self.channels_bounds.append(channel_bounds)
            self.channels_capacities.append(channel_capacities)

    def reverse_stego(self):
        self.result = bitarray()
        for channel in range(self.channels_count):
            for block in range(self.blocks_count):
                fv = self.channels_fvs[channel][block]
                l_bound, u_bound = self.channels_bounds[channel][block]
                n_bits = self.channels_capacities[channel][block]

                dec = abs(fv - l_bound)
                for _ in range(n_bits):
                    self.result.append(dec % 2)
                    dec >>= 1

        # remove padding
        initial_len = self.result
        while len(self.result) > 0 and self.result.pop() == 0:
            pass
        if len(self.result) == 0:
            # the message was all 0s
            self.result = bitarray([0] * initial_len)

        return self.result
    
    def extract(self, range_table):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(range_table)
        result = self.reverse_stego()
        return result
    
