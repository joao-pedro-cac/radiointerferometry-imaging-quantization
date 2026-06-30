"""
IEEE 754 Casting Library
===========

A library for analyzing and converting floating-point numbers through bitwise operations.

It includes
    1. functions to read the sign, exponent and mantissa field values of a floating-point number
    2. casting functions from higher-resolution float types to lower-resolution ones
"""
import struct

def get_double_datafields(value: float):
    """
    Description
    -----------

    Data field extraction function for double-precision
    floating-point numbers

    Input:
    ------
    - value: float64 value

    Output:
    -------
    - sign:     sign bit
    - exponent: 11-bit exponent field
    - mantissa: 52-bit mantissa field
    """

    double_bytes = struct.pack('d', value)
    uint_value = struct.unpack('Q', double_bytes)[0]

    # bit masks
    mantissa_mask = (1 << 52) - 1
    exponent_mask = ((1 << 11) - 1) << 52

    # read fields using bit masking
    mantissa = uint_value & mantissa_mask
    exponent = (uint_value & exponent_mask) >> 52
    sign = uint_value >> 63

    return sign, exponent, mantissa




def get_float_datafields(value: float):
    """
    Description
    -----------

    Data field extraction function for single-precision
    floating-point numbers

    Input:
    ------
    - value: float32 value

    Output:
    -------
    - sign:     sign bit
    - exponent: 8-bit exponent field
    - mantissa: 23-bit mantissa field
    """

    float_bytes = struct.pack('f', value)
    uint_value = struct.unpack('I', float_bytes)[0]

    # bit masks
    mantissa_mask = (1 << 23) - 1
    exponent_mask = ((1 << 8) - 1) << 23

    # read fields using bit masking
    mantissa = uint_value & mantissa_mask
    exponent = (uint_value & exponent_mask) >> 23
    sign = uint_value >> 31

    return sign, exponent, mantissa




def get_half_datafields(value: float):
    """
    Description
    -----------

    Data field extraction function for half-precision
    floating-point numbers

    Input:
    ------
    - value: float16 value

    Output:
    -------
    - sign:     sign bit
    - exponent: 5-bit exponent field
    - mantissa: 10-bit mantissa field
    """

    half_bytes = struct.pack('e', value)
    uint_value = struct.unpack('H', half_bytes)[0]

    # bit masks
    mantissa_mask = (1 << 10) - 1
    exponent_mask = ((1 << 5) - 1) << 10

    # read fields using bit masking
    mantissa = uint_value & mantissa_mask
    exponent = (uint_value & exponent_mask) >> 10
    sign = uint_value >> 15

    return sign, exponent, mantissa




def get_bfloat_datafields(value: float):
    """
    Description
    -----------

    Data field extraction function for 16-bit brain
    floating-point numbers

    Input:
    ------
    - value: bfloat16 value

    Output:
    -------
    - sign:     sign bit
    - exponent: 8-bit exponent field
    - mantissa: 7-bit mantissa field
    """

    bfloat_bytes = struct.pack('f', value)
    uint_value = struct.unpack('I', bfloat_bytes)[0]

    # bit masks
    mantissa_mask = (1 << 23) - 1
    exponent_mask = ((1 << 8) - 1) << 23

    # read fields using bit masking
    mantissa = (uint_value & mantissa_mask) >> 16
    exponent = (uint_value & exponent_mask) >> 23
    sign = uint_value >> 31

    return sign, exponent, mantissa




def double_to_float(value: float):
    """
    Description
    -----------

    Conversion function from double-precision
    to single-precision floating-point numbers
    using bit masking

    Input:
    --------
    - value: float64 value

    Output:
    --------
    - float_value: float32 number whose value is
                   (approximately) the same as the
                   input data
    - error: quantization (relative) error
    """

    double_sign, double_exponent, double_mantissa = get_double_datafields(value)

    # IEEE 754-defined biases
    double_bias = (1 << 10) - 1
    float_bias = (1 << 7) - 1

    float_exponent_mask = ((1 << 8) - 1)    # single-precision exponent bit mask

    float_sign = double_sign
    float_exponent = (double_exponent + double_bias - float_bias) & float_exponent_mask
    float_mantissa = (double_mantissa >> (52 - 23))
    # float_mantissa += 1 if double_mantissa >> (52 - 23 + 1) & 1 == 1 else 0    # conventional rounding (rather than truncation)

    uint_value = (float_sign << 31) + (float_exponent << 23) + float_mantissa  # IEEE 754-compliant layout

    float_value = struct.unpack('f', struct.pack('I', uint_value))[0]          # uint32 to float32 conversion
    error = abs(1 - float_value / value)                                       # quantization error

    return float_value, error




def float_to_half(value: float):
    """
    Description
    -----------

    Conversion function from single-precision
    to half-precision floating-point numbers
    using bit masking

    Input:
    ------
    - value: float32 value

    Output:
    -------
    - half_value: float16 number whose value is
                  (approximately) the same as the
                  input data
    - error: quantization (relative) error
    """

    # single-precision floating-point number
    float_sign, float_exponent, float_mantissa = get_float_datafields(value)

    # IEEE 754-defined biases
    float_bias = (1 << 7) - 1
    half_bias = (1 << 4) - 1
    # -----------------------------------------


    half_exponent_mask = (1 << 5) - 1

    half_sign = float_sign
    half_exponent = (float_exponent + float_bias - half_bias) & half_exponent_mask
    half_mantissa = (float_mantissa >> (23 - 10))
    # half_mantissa += 1 if float_mantissa >> (23 - 10 + 1) & 1 == 1 else 0    # conventional rounding (rather than truncation)

    uint_value = (half_sign << 15) + (half_exponent << 10) + half_mantissa   # IEEE 754-compliant layout

    half_value = struct.unpack('e', struct.pack('H', uint_value))[0]         # uint16 to float16 conversion
    error = abs(1 - half_value / value)                                      # quantization error

    return half_value, error




def float_to_bfloat(value: float):
    """
    Description
    -----------

    Conversion function from single-precision
    to (16-bit) brain floating-point numbers
    using bit masking

    Input:
    ------
    - value: float32 value

    Output:
    -------
    - bfloat_value: bfloat16 number whose value is
                    (approximately) the same as the
                    input data
    - error: quantization (relative) error
    """

    bfloat_mantissa_mask = ((1 << 16) - 1) << 16                            # only the mantissa mask is needed

    uint_value = struct.unpack('I', struct.pack('f', value))[0]

    new_uint_value = uint_value & bfloat_mantissa_mask
    # new_uint_value += 1 if uint_value & (1 << 15) == 1 else 0               # conventional rounding (rather than truncation)

    bfloat_value = struct.unpack('f', struct.pack('I', new_uint_value))[0]  # uint32 to float32 conversion
    error = abs(1 - bfloat_value / value)                                   # quantization error

    return bfloat_value, error




def double_to_half(value: float):
    """
    Description
    -----------

    Conversion function from double-precision
    to half-precision floating-point numbers
    using bit masking

    Input:
    ------
    - value: float64 value

    Output:
    -------
    - half_value: float16 number whose value is
                  (approximately) the same as the
                  input data
    - error: quantization (relative) error
    """
    float_value, _ = double_to_float(value)
    half_value, _ = float_to_half(float_value)
    error = abs(1 - half_value / value)

    return half_value, error




def double_to_bfloat(value: float):
    """
    Description
    -----------

    Conversion function from double-precision
    to (16-bit) brain floating-point numbers
    using bit masking

    Input:
    ------
    - value: float64 value


    Output:
    -------
    - bfloat_value: bfloat16 number whose value is
                    (approximately) the same as the
                    input data
    - error: quantization (relative) error
    """

    float_value, _ = double_to_float(value)
    bfloat_value, _ = float_to_bfloat(float_value)
    error = abs(1 - bfloat_value / value)

    return bfloat_value, error