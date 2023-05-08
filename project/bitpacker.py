from os.path import isdir
from os import remove, removedirs
import shutil
from bitarray import bitarray

def packSecrets(path = './secrets'):
    if isdir(path):
        shutil.make_archive('temp_file', 'zip', path)
        f = open('temp_file.zip', 'rb')
        bytes = f.read()
        f.close()
        remove('temp_file.zip')

        result = bitarray()
        result.frombytes(bytes)
        return result
    else:
        raise ValueError(f'{path} not a directory')
    

def unpackSecrets(bits: bitarray, out: str = './result_secrets'):
    shutil.rmtree(out)
    f = open('temp_file.zip', 'wb')
    f.write(bits.tobytes())
    f.close()
    shutil.unpack_archive('temp_file.zip', out, 'zip')
    remove('temp_file.zip')
