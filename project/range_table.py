# %%
class RangeTables:

    def basic_range_table(fv):
        if fv < 8: return 0, 7, 3
        elif fv < 16: return 8, 15, 3
        elif fv < 32: return 16, 31, 4
        elif fv < 64: return 32, 63, 5
        elif fv < 128: return 64, 127, 6
        else: return 128, 255, 7

    def custom_range_table(bit_counts, fv):
        if len(bit_counts) != 6:
            raise ValueError('A custom range tab;e must be created form 6 values')
        if fv < 8: return 0, 7, bit_counts[0]
        elif fv < 16: return 8, 15, bit_counts[1]
        elif fv < 32: return 16, 31, bit_counts[2]
        elif fv < 64: return 32, 63, bit_counts[3]
        elif fv < 128: return 64, 127, bit_counts[4]
        else: return 128, 255, bit_counts[5]

    def make_range_table(bit_counts):
        return lambda fv : RangeTables.custom_range_table(bit_counts, fv)

