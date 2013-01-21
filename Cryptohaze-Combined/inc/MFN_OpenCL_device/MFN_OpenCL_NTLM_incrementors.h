#define MFNSingleIncrementorsNTLMOpenCL1(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } 


#define MFNSingleIncrementorsNTLMOpenCL2(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } 


#define MFNSingleIncrementorsNTLMOpenCL3(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } 


#define MFNSingleIncrementorsNTLMOpenCL4(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } 


#define MFNSingleIncrementorsNTLMOpenCL5(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL6(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL7(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL8(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL9(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL10(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL11(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL12(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(b5.s##suffix >> 16) & 0xff]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL13(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(b5.s##suffix >> 16) & 0xff]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(b6.s##suffix >> 0) & 0xff]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL14(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(b5.s##suffix >> 16) & 0xff]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(b6.s##suffix >> 0) & 0xff]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b6.s##suffix >> 16) & 0xff]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL15(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(b5.s##suffix >> 16) & 0xff]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(b6.s##suffix >> 0) & 0xff]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b6.s##suffix >> 16) & 0xff]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                            passOffset = charsetReverse[(b7.s##suffix >> 0) & 0xff]; \
                            b7.s##suffix &= 0xffffff00;\
                            passOffset++;\
                            b7.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                            if (passOffset >= charsetLengths[0]) { \
                              b7.s##suffix &= 0xffffff00;\
                              b7.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } } } 


#define MFNSingleIncrementorsNTLMOpenCL16(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(b4.s##suffix >> 0) & 0xff]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b4.s##suffix >> 16) & 0xff]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(b5.s##suffix >> 0) & 0xff]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(b5.s##suffix >> 16) & 0xff]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(b6.s##suffix >> 0) & 0xff]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b6.s##suffix >> 16) & 0xff]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                            passOffset = charsetReverse[(b7.s##suffix >> 0) & 0xff]; \
                            b7.s##suffix &= 0xffffff00;\
                            passOffset++;\
                            b7.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                            if (passOffset >= charsetLengths[0]) { \
                              b7.s##suffix &= 0xffffff00;\
                              b7.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                              passOffset = charsetReverse[(b7.s##suffix >> 16) & 0xff]; \
                              b7.s##suffix &= 0xff00ffff;\
                              passOffset++;\
                              b7.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                              if (passOffset >= charsetLengths[0]) { \
                                b7.s##suffix &= 0xff00ffff;\
                                b7.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL1(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
} } 


#define MFNMultipleIncrementorsNTLMOpenCL2(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
} } } 


#define MFNMultipleIncrementorsNTLMOpenCL3(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
} } } } 


#define MFNMultipleIncrementorsNTLMOpenCL4(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
} } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL5(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
} } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL6(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
} } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL7(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
} } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL8(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
} } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL9(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
} } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL10(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
} } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL11(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
} } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL12(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + (b5.s##suffix >> 16) & 0xff)]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 16);\
} } } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL13(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + (b5.s##suffix >> 16) & 0xff)]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + (b6.s##suffix >> 0) & 0xff)]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
} } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL14(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + (b5.s##suffix >> 16) & 0xff)]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + (b6.s##suffix >> 0) & 0xff)]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b6.s##suffix >> 16) & 0xff)]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 16);\
} } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL15(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + (b5.s##suffix >> 16) & 0xff)]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + (b6.s##suffix >> 0) & 0xff)]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b6.s##suffix >> 16) & 0xff)]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 16);\
                            passOffset = charsetReverse[((128 * 14) + (b7.s##suffix >> 0) & 0xff)]; \
                            b7.s##suffix &= 0xffffff00;\
                            passOffset++;\
                            b7.s##suffix |= (unsigned int)(charsetForward[(128 * 14) + passOffset] << 0);\
                            if (passOffset >= charsetLengths[14]) { \
                              b7.s##suffix &= 0xffffff00;\
                              b7.s##suffix |= (unsigned int)(charsetForward[(128 * 14)] << 0);\
} } } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsNTLMOpenCL16(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 16) & 0xff)]; \
  b0.s##suffix &= 0xff00ffff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xff00ffff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + (b1.s##suffix >> 0) & 0xff)]; \
    b1.s##suffix &= 0xffffff00;\
    passOffset++;\
    b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      b1.s##suffix &= 0xffffff00;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + (b1.s##suffix >> 16) & 0xff)]; \
      b1.s##suffix &= 0xff00ffff;\
      passOffset++;\
      b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        b1.s##suffix &= 0xff00ffff;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + (b2.s##suffix >> 0) & 0xff)]; \
        b2.s##suffix &= 0xffffff00;\
        passOffset++;\
        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b2.s##suffix &= 0xffffff00;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b2.s##suffix >> 16) & 0xff)]; \
          b2.s##suffix &= 0xff00ffff;\
          passOffset++;\
          b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            b2.s##suffix &= 0xff00ffff;\
            b2.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + (b3.s##suffix >> 0) & 0xff)]; \
            b3.s##suffix &= 0xffffff00;\
            passOffset++;\
            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              b3.s##suffix &= 0xffffff00;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + (b3.s##suffix >> 16) & 0xff)]; \
              b3.s##suffix &= 0xff00ffff;\
              passOffset++;\
              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                b3.s##suffix &= 0xff00ffff;\
                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + (b4.s##suffix >> 0) & 0xff)]; \
                b4.s##suffix &= 0xffffff00;\
                passOffset++;\
                b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b4.s##suffix &= 0xffffff00;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b4.s##suffix >> 16) & 0xff)]; \
                  b4.s##suffix &= 0xff00ffff;\
                  passOffset++;\
                  b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    b4.s##suffix &= 0xff00ffff;\
                    b4.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + (b5.s##suffix >> 0) & 0xff)]; \
                    b5.s##suffix &= 0xffffff00;\
                    passOffset++;\
                    b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      b5.s##suffix &= 0xffffff00;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + (b5.s##suffix >> 16) & 0xff)]; \
                      b5.s##suffix &= 0xff00ffff;\
                      passOffset++;\
                      b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        b5.s##suffix &= 0xff00ffff;\
                        b5.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + (b6.s##suffix >> 0) & 0xff)]; \
                        b6.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b6.s##suffix &= 0xffffff00;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b6.s##suffix >> 16) & 0xff)]; \
                          b6.s##suffix &= 0xff00ffff;\
                          passOffset++;\
                          b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            b6.s##suffix &= 0xff00ffff;\
                            b6.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 16);\
                            passOffset = charsetReverse[((128 * 14) + (b7.s##suffix >> 0) & 0xff)]; \
                            b7.s##suffix &= 0xffffff00;\
                            passOffset++;\
                            b7.s##suffix |= (unsigned int)(charsetForward[(128 * 14) + passOffset] << 0);\
                            if (passOffset >= charsetLengths[14]) { \
                              b7.s##suffix &= 0xffffff00;\
                              b7.s##suffix |= (unsigned int)(charsetForward[(128 * 14)] << 0);\
                              passOffset = charsetReverse[((128 * 15) + (b7.s##suffix >> 16) & 0xff)]; \
                              b7.s##suffix &= 0xff00ffff;\
                              passOffset++;\
                              b7.s##suffix |= (unsigned int)(charsetForward[(128 * 15) + passOffset] << 16);\
                              if (passOffset >= charsetLengths[15]) { \
                                b7.s##suffix &= 0xff00ffff;\
                                b7.s##suffix |= (unsigned int)(charsetForward[(128 * 15)] << 16);\
} } } } } } } } } } } } } } } } } 


