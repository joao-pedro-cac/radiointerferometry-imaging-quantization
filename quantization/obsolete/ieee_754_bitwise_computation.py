import ctypes


class DataUnion64Bits(ctypes.Union):
    """
    C-like `union` structure for
    64-bit numeric data types
    """
    _fields_ = [
        ("int",    ctypes.c_int64),
        ("uint",   ctypes.c_uint64),
        ("double", ctypes.c_double),
        ("floats", ctypes.c_float * 2)
    ]


class DataUnion32Bits(ctypes.Union):
    """
    C-like `union` structure for
    32-bit numeric data types
    """
    _fields_ = [
        ("int",   ctypes.c_int32),
        ("uint",  ctypes.c_uint32),
        ("float", ctypes.c_float),
    ]


def get_double_datafields(value: DataUnion64Bits):
    """
    Description
    -----------

    Data field extraction function for double-precision
    floating-point numbers

    Input:
    ------
    - value: DataUnion64Bits object

    Output:
    -------
    - sign:     sign bit
    - exponent: 11-bit exponent field
    - mantissa: 52-bit mantissa field
    """

    assert type(value) == DataUnion64Bits    # verify that the data is encoded on 64 bits

    # bit masks
    mantissa_mask = (1 << 52) - 1
    exponent_mask = ((1 << 11) - 1) << 52

    # read fields using bit masking
    mantissa = value.uint & mantissa_mask
    exponent = (value.uint & exponent_mask) >> 52
    sign = value.uint >> 63

    return sign, exponent, mantissa


def get_float_datafields(value: DataUnion32Bits):
    """
    Description
    -----------

    Data field extraction function for single-precision
    floating-point numbers

    Input:
    ------
    - value: DataUnion32Bits object

    Output:
    -------
    - sign:     sign bit
    - exponent: 8-bit exponent field
    - mantissa: 23-bit mantissa field
    """

    assert type(value) == DataUnion32Bits    # verify that the data is encoded on 32 bits

    # bit masks
    mantissa_mask = (1 << 23) - 1
    exponent_mask = ((1 << 8) - 1) << 23

    # read fields using bit masking
    mantissa = value.uint & mantissa_mask
    exponent = (value.uint & exponent_mask) >> 23
    sign = value.uint >> 31

    return sign, exponent, mantissa


def get_bfloat_datafields(value: DataUnion32Bits):
    """
    Description
    -----------

    Data field extraction function for 16-bit brain
    floating-point numbers

    Input:
    ------
    - value: DataUnion32Bits object

    Output:
    -------
    - sign:     sign bit
    - exponent: 8-bit exponent field
    - mantissa: 7-bit mantissa field
    """

    assert type(value) == DataUnion32Bits    # verify that the data is encoded on 32 bits

    # bit masks
    mantissa_mask = (1 << 23) - 1
    exponent_mask = ((1 << 8) - 1) << 23

    # read fields using bit masking
    mantissa = (value.uint & mantissa_mask) >> 16
    mantissa += 1 if ((value.uint & mantissa_mask) >> (16 - 1)) & 1 == 1 else 0
    exponent = (value.uint & exponent_mask) >> 23
    sign = value.uint >> 31

    return sign, exponent, mantissa


def double_to_float(un64: DataUnion64Bits):
    """
    Description
    -----------

    Conversion function from double-precision
    to single-precision floating-point numbers
    using bit masking

    Input:
    ------
    - un64: DataUnion64Bits object


    Output:
    -------
    - un32: DataUnion32Bits object whose value
            is (approximately) the same as that
            of the input data
    - error: Quantization (relative) error
    """

    assert type(un64) == DataUnion64Bits    # verify that the data is encoded on 64 bits

    # single-precision floating-point number
    un32 = DataUnion32Bits()
    double_sign, double_exponent, double_mantissa = get_double_datafields(un64)

    # IEEE 754-defined biases
    double_bias = (1 << 10) - 1
    float_bias = (1 << 7) - 1

    float_exponent_mask = ((1 << 8) - 1)    # single-precision exponent bit mask

    float_sign = double_sign
    float_exponent = (double_exponent + double_bias - float_bias) & float_exponent_mask
    float_mantissa = (double_mantissa >> (52 - 23))
    float_mantissa += 1 if double_mantissa >> (52 - 23 + 1) & 1 == 1 else 0    # conventional rounding (rather than truncation)

    un32.uint = (float_sign << 31) + (float_exponent << 23) + float_mantissa   # IEEE 754-compliant layout

    error = abs(1 - un32.float / un64.double)

    return un32, error


def float_to_bfloat(un32: DataUnion32Bits):
    """
    Description
    -----------

    Conversion function from single-precision
    to (16-bit) brain floating-point numbers
    using bit masking

    Input:
    ------
    - un32: DataUnion32Bits object


    Output:
    -------
    - un32: DataUnion32Bits object whose value
            is (approximately) the same as that
            of the input data
    - error: Quantization (relative) error
    """

    assert type(un32) == DataUnion32Bits    # verify that the data is encoded on 32 bits

    bfloat_mask = ((1 << 16) - 1) << 16

    un32brain = DataUnion32Bits()
    un32brain.uint = un32.uint & bfloat_mask

    error = abs(1 - un32brain.float / un32.float)

    return un32brain, error