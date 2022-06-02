import os
import shutil
import zipfile
import gzip



class CompressedFile(object):
    magic = None
    file_type = None
    mime_type = None
    def __init__(self, fname, save_dir):
        self.extract_all(fname, save_dir)

    @classmethod
    def is_magic(self, data):
        return data.startswith(self.magic)


    def extract_all(self, fname, save_dir):
        pass


class ZIPFile (CompressedFile):
    magic = b'\x50\x4b\x03\x04'
    file_type = 'zip'
    mime_type = 'compressed/zip'


    def extract_all(self, fname, save_dir):
        with zipfile.ZipFile(fname) as f:
            for member in f.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
            
                source = f.open(member)
                target = open(os.path.join(save_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)



class GZFile (CompressedFile):
    magic = b'\x1f\x8b\x08'
    file_type = 'gz'
    mime_type = 'compressed/gz'

    
    def extract_all(self, fname, save_dir):
        decompressed_fname = os.path.basename(fname)[:-3]
        with gzip.open(fname, 'rb') as f_in:
            with open(os.path.join(save_dir, decompressed_fname), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def extract_compressed_file(filename, save_dir):
    with open(filename, 'rb') as f:
        start_of_file = f.read(1024)
        
        f.seek(0)
        if filename.endswith('csv'):
            basename = os.path.basename(filename)
            with open(os.path.join(save_dir, basename), 'wb') as f_out:
                shutil.copyfileobj(f, f_out)
        else:
            for cls in (ZIPFile, GZFile):
                if cls.is_magic(start_of_file):
                    cls(filename, save_dir)
                    break


