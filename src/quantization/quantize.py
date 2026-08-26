import numpy as np
import quantization.ieee_754_casting as ieee_casting


def quantize_image(image, quantization_type, float16_rescale_max=65503):
    assert float16_rescale_max < 65504
    scale_factor_img = 1

    if quantization_type == "float64":
        image = image.astype(np.float64)
    else:
        image = image.astype(np.float32)

        if quantization_type == "float16":
            scale_factor_img = float16_rescale_max / np.max(np.abs(image))
            image = (image * scale_factor_img).astype(np.float32)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i, j] = ieee_casting.float_to_half(image[i, j])[0]

            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        elif quantization_type == "bfloat16":
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i, j] = ieee_casting.float_to_bfloat(image[i, j])[0]

    return image, scale_factor_img




def quantize_weights(wgt, quantization_type, float16_rescale_max=65503):
    assert float16_rescale_max < 65504
    scale_factor_wgt = 1

    if quantization_type == "float64":
        wgt = wgt.astype("float64")
    else:
        wgt = wgt.astype("float32")

        if quantization_type == "float16":
            scale_factor_wgt = float16_rescale_max / np.sum(wgt)
            wgt = (wgt * scale_factor_wgt).astype(np.float32)

            for i in range(wgt.shape[0]):
                for j in range(wgt.shape[1]):
                    wgt[i, j] = ieee_casting.float_to_half(wgt[i, j])[0]

            wgt = np.nan_to_num(wgt, nan=0.0, posinf=0.0, neginf=0.0)

        elif quantization_type == "bfloat16":
            for i in range(wgt.shape[0]):
                for j in range(wgt.shape[1]):
                    wgt[i, j] = ieee_casting.float_to_bfloat(wgt[i, j])[0]

    return wgt, scale_factor_wgt




def quantize_visibilities(vis, quantization_type, float16_rescale_max=65503, conserve_scalefactor=False, scale_factor_vis=1):
    assert float16_rescale_max < 65504

    if quantization_type == "float64":
        vis = vis.astype("complex128")
    else:
        vis = vis.astype("complex64")

        if quantization_type == "float16":
            if not conserve_scalefactor:
                scale_factor_vis = float16_rescale_max / np.max(np.abs(vis))
            vis = (vis * scale_factor_vis).astype(np.complex64)

            for i in range(vis.shape[0]):
                for j in range(vis.shape[1]):
                    data = complex(vis[i, j])
                    vis[i, j] = ieee_casting.float_to_half(data.real)[0] + ieee_casting.float_to_half(data.imag)[0] * 1j

        elif quantization_type == "bfloat16":
            for i in range(vis.shape[0]):
                for j in range(vis.shape[1]):
                    data = complex(vis[i, j])
                    vis[i, j] = ieee_casting.float_to_bfloat(data.real)[0] + ieee_casting.float_to_bfloat(data.imag)[0] * 1j

    return vis, scale_factor_vis



