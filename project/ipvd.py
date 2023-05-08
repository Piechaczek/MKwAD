from PIL import Image
from math import ceil, floor
import numpy as np
from range_table import RangeTables
from bitarray import bitarray
from copy import copy

class IPVD:
    
    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)
        self.blocks_count: int = 0 # Initialized after calculating blocks

        self.channels_blocks = None # Fluctuation values (differences between pixels)
        self.channels_blocks_coords = None # Row an column of teh first pixel in the block
        self.channels_fvs = None # Fluctuation values (differences between pixels)
        self.channels_bounds = None # (Lower bound, Upper bound) for every block
        self.channels_capacities = None # Capacity (n_bits) for every block
        self.stego_result = None # Resulting pixels of the stego image (list of lists of pixels)
        self.flattened_stego = None # Resulting pixels of the stego image (list of color tuples)
        self.calculate_blocks() # Init the lists above

    def calculate_blocks(self):
        self.channels_blocks = []
        self.channels_blocks_coords = []
        self.channels_fvs = []
        for channel in self.channels_px:
            channel_blocks = []
            channel_blocks_coords = []
            channel_fvs = []
            for r in range(self.im.height):
                for c in range(self.im.width):
                    if c % 3 == 0 and r % 2 == 0 and r + 1 < self.im.height and c + 2 < self.im.width:
                        # get pixels from 2x3 block
                        # print(r, c, self.im.width, self.im.height, len(channel))
                        px0 = channel[r * self.im.width + c]
                        px1 = channel[r * self.im.width + c + 1]
                        px2 = channel[(r + 1) * self.im.width + c]
                        px3 = channel[(r + 1) * self.im.width + c + 1]
                        px4 = channel[r * self.im.width + c + 2]
                        px5 = channel[(r + 1) * self.im.width + c + 2]
                        block_px = (px0, px1, px2, px3, px4, px5)
                        channel_blocks.append(block_px)
                        channel_blocks_coords.append((r, c))

                        # calc difference
                        f1 = abs(px0 - px3)
                        f2 = abs(px1 - px2)
                        channel_fvs.append((f1, f2))
            self.channels_blocks.append(channel_blocks)
            self.channels_blocks_coords.append(channel_blocks_coords)
            self.channels_fvs.append(channel_fvs)
        self.blocks_count = len(self.channels_blocks[0]) # All channels have the same amount of blocks, take the first channel's blocks count

    def calculate_ranges(self, range_table, k=2):
        self.channels_bounds = []
        self.channels_capacities = []
        for channel in range(self.channels_count):
            channel_bounds = []
            channel_capacities = []
            for block in range(self.blocks_count):
                # range table lookups
                f1, f2 = self.channels_fvs[channel][block]
                l_bound1, u_bound1, n_bits1 = range_table(f1)
                l_bound2, u_bound2, n_bits2 = range_table(f2)
                n_bits_indicator = k
                channel_bounds.append((l_bound1, u_bound1, l_bound2, u_bound2))
                channel_capacities.append((n_bits1, n_bits2, n_bits_indicator))
            self.channels_bounds.append(channel_bounds)
            self.channels_capacities.append(channel_capacities)

    def get_capacity(self):
        if self.channels_capacities is None:
            raise RuntimeError('calculate_ranges must be called before capacity can be calculated!')
        sum = 0
        for channel in range(self.channels_count):
            for block in range(self.blocks_count):
                n1, n2, n_indicator = self.channels_capacities[channel][block]
                sum += n1 + n2 + 2 * n_indicator
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
            result = copy(list(self.channels_px[channel]))
            for block in range(self.blocks_count):
                px0, px1, px2, px3, px4, px5 = self.channels_blocks[channel][block]
                f1, f2 = self.channels_fvs[channel][block]
                n1_bits, n2_bits, n_bits_indicator = self.channels_capacities[channel][block]
                l1_bound, u1_bound, l2_bound, u2_bound = self.channels_bounds[channel][block]


                # get the decimal values
                dec1 = 0
                for j in range(n1_bits):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    dec1 += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1

                dec2 = 0
                for j in range(n2_bits):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    dec2 += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1
                
                # new differences
                f1_new = l1_bound + floor(dec1 / 2)
                f2_new = l2_bound + floor(dec2 / 2) # TODO maybe should be ceil?
                diff1 = abs(f1 - f1_new)
                diff2 = abs(f2 - f2_new)

                # calc new pixels from equations
                if px0 >= px3 and f1_new > f1:
                    px0_new = px0 + ceil(diff1 / 2)
                    px3_new = px3 - floor(diff1 / 2)
                elif px0 < px3 and f1_new > f1:
                    px0_new = px0 - floor(diff1 / 2)
                    px3_new = px3 + ceil(diff1 / 2)
                elif px0 >= px3 and f1_new <= f1:
                    px0_new = px0 - ceil(diff1 / 2)
                    px3_new = px3 + floor(diff1 / 2)
                else:
                    px0_new = px0 + ceil(diff1 / 2)
                    px3_new = px3 - floor(diff1 / 2)

                if px1 >= px2 and f2_new > f2:
                    px1_new = px1 + ceil(diff2 / 2)
                    px2_new = px2 - floor(diff2 / 2)
                elif px1 < px2 and f2_new > f2:
                    px1_new = px1 - floor(diff2 / 2)
                    px2_new = px2 + ceil(diff2 / 2)
                elif px1 >= px2 and f2_new <= f2:
                    px1_new = px1 - ceil(diff2 / 2)
                    px2_new = px2 + floor(diff2 / 2)
                else:
                    px1_new = px1 + ceil(diff2 / 2)
                    px2_new = px2 - floor(diff2 / 2)
                
                # indicator pixels
                if dec1 % 2 == px4 % 2:
                    px4_new = px4
                elif dec1 % 2 == 0:
                    px4_new = px4 - 1
                else:
                    px4_new = px4 + 1

                if dec2 % 2 == px5 % 2:
                    px5_new = px5
                elif dec2 % 2 == 0:
                    px5_new = px5 - 1
                else:
                    px5_new = px5 + 1

                # LSB (starting from second-last bit)
                # get the decimal values
                dec3 = 0
                for j in range(n_bits_indicator):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    dec3 += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1

                dec4 = 0
                for j in range(n_bits_indicator):
                    bit_to_embed, data_ptr_delta = self.get_bit_to_embed(data, data_ptr, result_ptr, capacity)
                    dec4 += (2 ** j) * bit_to_embed
                    data_ptr += data_ptr_delta
                    result_ptr += 1

                # calculate deltas
                delta1 = (n_bits_indicator**2 - 1) - dec3
                delta2 = (n_bits_indicator**2 - 1) - dec4

                # embed the deltas
                # the new pixel values are at most 9 bits long, by my calculations
                px4_new &= 0b111111001 
                px5_new &= 0b111111001 
                px4_new |= delta1 << 1
                px5_new |= delta2 << 1

                # Boundry issues (BUP)
                if px0_new < 0:
                    px3_new -= px0_new
                    px0_new = 0
                if px3_new < 0:
                    px0_new -= px3_new
                    px3_new = 0
                if px1_new < 0:
                    px2_new -= px1_new
                    px1_new = 0
                if px2_new < 0:
                    px1_new -= px2_new
                    px2_new = 0

                # Boundry issues (BOP)
                if px0_new > 255:
                    px3_new += px0_new - 255
                    px0_new = 255
                if px3_new > 255:
                    px0_new -= px3_new - 255
                    px3_new = 255
                if px1_new > 255:
                    px2_new -= px1_new - 255
                    px1_new = 255
                if px2_new > 255:
                    px1_new -= px2_new - 255
                    px2_new = 255

                # Save result
                # print('n', n_bits, 'data', embedded_num, 'px', px1, px2, 'fv', fv, fv_new, 'px_new', px1_new, px2_new) 
                r, c = self.channels_blocks_coords[channel][block]
                result[r * self.im.width + c] = px0_new
                result[r * self.im.width + c + 1] = px1_new
                result[(r + 1) * self.im.width + c] = px2_new
                result[(r + 1) * self.im.width + c + 1] = px3_new
                result[r * self.im.width + c + 2] = px4_new
                result[(r + 1) * self.im.width + c + 2] = px5_new
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

    def hide_data(self, data: bytearray, range_table, k, out_name: str):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(range_table, k)
        new_data = self.calculate_stego(data)
        result = self.get_result(out_name)
        return result, new_data


class ReverseIPVD:

    def __init__(self, image_name: str) -> None:
        # Process image
        self.im: Image.Image = Image.open(image_name).convert('RGBA') # Pillow image
        self.channels_px = list(map(lambda channel : channel.getdata(), self.im.split())) # Individual channels' pixel values

        self.channels_count: int = len(self.channels_px)
        self.blocks_count: int = 0 # Initialized after calculating blocks

        self.channels_blocks = None # Fluctuation values (differences between pixels)
        self.channels_blocks_coords = None # Row an column of teh first pixel in the block
        self.channels_fvs = None # Fluctuation values (differences between pixels)
        self.channels_bounds = None # (Lower bound, Upper bound) for every block
        self.channels_capacities = None # Capacity (n_bits) for every block
        self.result = [] # Resulting pixels of the stego image
        self.calculate_blocks() # Init the lists above

    def calculate_blocks(self):
        self.channels_blocks = []
        self.channels_blocks_coords = []
        self.channels_fvs = []
        for channel in self.channels_px:
            channel_blocks = []
            channel_blocks_coords = []
            channel_fvs = []
            for r in range(self.im.height):
                for c in range(self.im.width):
                    if c % 3 == 0 and r % 2 == 0 and r + 1 < self.im.height and c + 2 < self.im.width:
                        # get pixels from 2x3 block
                        # print(r, c, self.im.width, self.im.height, len(channel))
                        px0 = channel[r * self.im.width + c]
                        px1 = channel[r * self.im.width + c + 1]
                        px2 = channel[(r + 1) * self.im.width + c]
                        px3 = channel[(r + 1) * self.im.width + c + 1]
                        px4 = channel[r * self.im.width + c + 2]
                        px5 = channel[(r + 1) * self.im.width + c + 2]
                        block_px = (px0, px1, px2, px3, px4, px5)
                        channel_blocks.append(block_px)
                        channel_blocks_coords.append((r, c))

                        # calc difference
                        f1 = abs(px0 - px3)
                        f2 = abs(px1 - px2)
                        channel_fvs.append((f1, f2))
            self.channels_blocks.append(channel_blocks)
            self.channels_blocks_coords.append(channel_blocks_coords)
            self.channels_fvs.append(channel_fvs)
        self.blocks_count = len(self.channels_blocks[0]) # All channels have the same amount of blocks, take the first channel's blocks count

    def calculate_ranges(self, range_table, k=2):
        self.channels_bounds = []
        self.channels_capacities = []
        for channel in range(self.channels_count):
            channel_bounds = []
            channel_capacities = []
            for block in range(self.blocks_count):
                # range table lookups
                f1, f2 = self.channels_fvs[channel][block]
                l_bound1, u_bound1, n_bits1 = range_table(f1)
                l_bound2, u_bound2, n_bits2 = range_table(f2)
                n_bits_indicator = k
                channel_bounds.append((l_bound1, u_bound1, l_bound2, u_bound2))
                channel_capacities.append((n_bits1, n_bits2, n_bits_indicator))
            self.channels_bounds.append(channel_bounds)
            self.channels_capacities.append(channel_capacities)

    def reverse_stego(self):
        self.result = bitarray()
        for channel in range(self.channels_count):
            for block in range(self.blocks_count):
                p0, p1, p2, p3, p4, p5 = self.channels_blocks[channel][block]
                f1, f2 = self.channels_fvs[channel][block]
                l1_bound, u1_bound, l2_bound, u2_bound = self.channels_bounds[channel][block]
                n1_bits, n2_bits, n_indicator = self.channels_capacities[channel][block]

                # extract
                dec1 = abs(f1 - l1_bound)
                dec2 = abs(f2 - l2_bound)

                # adjust with indicators
                dec1 = (dec1 << 1) | p4 % 2
                dec2 = (dec2 << 1) | p5 % 2

                # get the LSB bits
                dec3 = (p4 & 0b110) >> 1
                dec4 = (p5 & 0b110) >> 1
                delta3 = (n_indicator**2 - 1) - dec3
                delta4 = (n_indicator**2 - 1) - dec4

                # append bits to result
                for _ in range(n1_bits):
                    self.result.append(dec1 % 2)
                    dec1 >>= 1

                # My modification, changed the order, the one in the paper made no sense
                for _ in range(n2_bits):
                    self.result.append(dec2 % 2)
                    dec2 >>= 1

                for _ in range(n_indicator):
                    self.result.append(delta3 % 2)
                    delta3 >>= 1

                for _ in range(n_indicator):
                    self.result.append(delta4 % 2)
                    delta4 >>= 1

        # remove padding
        initial_len = self.result
        while len(self.result) > 0 and self.result.pop() == 0:
            pass
        if len(self.result) == 0:
            # the message was all 0s
            self.result = bitarray([0] * initial_len)

        return self.result
    
    def extract(self, range_table, k):
        # convenience function combining all of the above into one pipeline
        self.calculate_ranges(range_table, k)
        result = self.reverse_stego()
        return result
