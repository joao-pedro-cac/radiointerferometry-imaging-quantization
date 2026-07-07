def extract_pixel_info(truth_model_datapath):
    with open(truth_model_datapath, 'rt') as fd:
        text = fd.read(5000).replace(' ', '')

        # indexes for finding image size and pixel size information
        naxis1_index = text.index('NAXIS1=') + len('NAXIS1=')
        naxis2_index = text.index('NAXIS2=') + len('NAXIS2=')
        ang1_deg_index = text.index('CDELT1=-0.') + len('CDELT1=-0.')
        ang2_deg_index = text.index('CDELT2=0.') + len('CDELT2=0.')

        # numerical variables
        val1 = ''
        val2 = ''
        ang1_deg = '0.'
        ang2_deg = '0.'


        # searching for image dimensions
        while text[naxis1_index].isascii() and text[naxis1_index].isdecimal():
            val1 += text[naxis1_index]
            naxis1_index += 1
        val1 = int(val1)
        
        while text[naxis2_index].isascii() and text[naxis2_index].isdecimal():
            val2 += text[naxis2_index]
            naxis2_index += 1
        val2 = int(val2)


        # searching for pixel sizes (in degrees)
        while text[ang1_deg_index].isascii() and text[ang1_deg_index].isdecimal():
            ang1_deg += text[ang1_deg_index]
            ang1_deg_index += 1
        ang1_deg = float(ang1_deg)
        
        while text[ang2_deg_index].isascii() and text[ang2_deg_index].isdecimal():
            ang2_deg += text[ang2_deg_index]
            ang2_deg_index += 1
        ang2_deg = float(ang2_deg)

        return val1, val2, ang1_deg, ang2_deg
    
def extract_telescope_file(file_path):
    with open(file_path, "r") as fd:
        file_text = fd.read()

        start_index = file_text.index("TELESCOPE=\"") + len("TELESCOPE=\"")

        end_index = start_index

        while file_text[end_index] != '\"':
            end_index += 1

        return file_text[start_index:end_index]