#define MFNSingleIncrementorsOpenCL1(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } 


#define MFNSingleIncrementorsOpenCL2(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
} } } 


#define MFNSingleIncrementorsOpenCL3(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } 


#define MFNSingleIncrementorsOpenCL4(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
} } } } } 


#define MFNSingleIncrementorsOpenCL5(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } 


#define MFNSingleIncrementorsOpenCL6(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
} } } } } } } 


#define MFNSingleIncrementorsOpenCL7(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } 


#define MFNSingleIncrementorsOpenCL8(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
} } } } } } } } } 


#define MFNSingleIncrementorsOpenCL9(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL10(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
} } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL11(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL12(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(b2.s##suffix >> 24) & 0xff]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
} } } } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL13(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(b2.s##suffix >> 24) & 0xff]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL14(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(b2.s##suffix >> 24) & 0xff]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b3.s##suffix >> 8) & 0xff]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
} } } } } } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL15(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(b2.s##suffix >> 24) & 0xff]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b3.s##suffix >> 8) & 0xff]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                            passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
                            b3.s##suffix &= 0xff00ffff;\
                            passOffset++;\
                            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                            if (passOffset >= charsetLengths[0]) { \
                              b3.s##suffix &= 0xff00ffff;\
                              b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } } 


#define MFNSingleIncrementorsOpenCL16(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[(b0.s##suffix >> 0) & 0xff]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(b0.s##suffix >> 8) & 0xff]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(b0.s##suffix >> 16) & 0xff]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(b0.s##suffix >> 24) & 0xff]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(b1.s##suffix >> 0) & 0xff]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(b1.s##suffix >> 8) & 0xff]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(b1.s##suffix >> 16) & 0xff]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(b1.s##suffix >> 24) & 0xff]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(b2.s##suffix >> 0) & 0xff]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(b2.s##suffix >> 8) & 0xff]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(b2.s##suffix >> 16) & 0xff]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(b2.s##suffix >> 24) & 0xff]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(b3.s##suffix >> 0) & 0xff]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(b3.s##suffix >> 8) & 0xff]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[0] << 8);\
                            passOffset = charsetReverse[(b3.s##suffix >> 16) & 0xff]; \
                            b3.s##suffix &= 0xff00ffff;\
                            passOffset++;\
                            b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 16);\
                            if (passOffset >= charsetLengths[0]) { \
                              b3.s##suffix &= 0xff00ffff;\
                              b3.s##suffix |= (unsigned int)(charsetForward[0] << 16);\
                              passOffset = charsetReverse[(b3.s##suffix >> 24) & 0xff]; \
                              b3.s##suffix &= 0x00ffffff;\
                              passOffset++;\
                              b3.s##suffix |= (unsigned int)(charsetForward[passOffset] << 24);\
                              if (passOffset >= charsetLengths[0]) { \
                                b3.s##suffix &= 0x00ffffff;\
                                b3.s##suffix |= (unsigned int)(charsetForward[0] << 24);\
} } } } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL1(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
} } 


#define MFNMultipleIncrementorsOpenCL2(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
} } } 


#define MFNMultipleIncrementorsOpenCL3(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
} } } } 


#define MFNMultipleIncrementorsOpenCL4(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
} } } } } 


#define MFNMultipleIncrementorsOpenCL5(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
} } } } } } 


#define MFNMultipleIncrementorsOpenCL6(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
} } } } } } } 


#define MFNMultipleIncrementorsOpenCL7(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
} } } } } } } } 


#define MFNMultipleIncrementorsOpenCL8(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
} } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL9(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
} } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL10(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
} } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL11(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
} } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL12(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + (b2.s##suffix >> 24) & 0xff)]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 24);\
} } } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL13(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + (b2.s##suffix >> 24) & 0xff)]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + (b3.s##suffix >> 0) & 0xff)]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
} } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL14(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + (b2.s##suffix >> 24) & 0xff)]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + (b3.s##suffix >> 0) & 0xff)]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b3.s##suffix >> 8) & 0xff)]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 8);\
} } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL15(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + (b2.s##suffix >> 24) & 0xff)]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + (b3.s##suffix >> 0) & 0xff)]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b3.s##suffix >> 8) & 0xff)]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 8);\
                            passOffset = charsetReverse[((128 * 14) + (b3.s##suffix >> 16) & 0xff)]; \
                            b3.s##suffix &= 0xff00ffff;\
                            passOffset++;\
                            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 14) + passOffset] << 16);\
                            if (passOffset >= charsetLengths[14]) { \
                              b3.s##suffix &= 0xff00ffff;\
                              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 14)] << 16);\
} } } } } } } } } } } } } } } } 


#define MFNMultipleIncrementorsOpenCL16(charsetForward, charsetReverse, charsetLengths, suffix) {\
passOffset = charsetReverse[((128 * 0) + (b0.s##suffix >> 0) & 0xff)]; \
b0.s##suffix &= 0xffffff00;\
passOffset++;\
b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  b0.s##suffix &= 0xffffff00;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + (b0.s##suffix >> 8) & 0xff)]; \
  b0.s##suffix &= 0xffff00ff;\
  passOffset++;\
  b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    b0.s##suffix &= 0xffff00ff;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + (b0.s##suffix >> 16) & 0xff)]; \
    b0.s##suffix &= 0xff00ffff;\
    passOffset++;\
    b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      b0.s##suffix &= 0xff00ffff;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + (b0.s##suffix >> 24) & 0xff)]; \
      b0.s##suffix &= 0x00ffffff;\
      passOffset++;\
      b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        b0.s##suffix &= 0x00ffffff;\
        b0.s##suffix |= (unsigned int)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + (b1.s##suffix >> 0) & 0xff)]; \
        b1.s##suffix &= 0xffffff00;\
        passOffset++;\
        b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          b1.s##suffix &= 0xffffff00;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + (b1.s##suffix >> 8) & 0xff)]; \
          b1.s##suffix &= 0xffff00ff;\
          passOffset++;\
          b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            b1.s##suffix &= 0xffff00ff;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + (b1.s##suffix >> 16) & 0xff)]; \
            b1.s##suffix &= 0xff00ffff;\
            passOffset++;\
            b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              b1.s##suffix &= 0xff00ffff;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + (b1.s##suffix >> 24) & 0xff)]; \
              b1.s##suffix &= 0x00ffffff;\
              passOffset++;\
              b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                b1.s##suffix &= 0x00ffffff;\
                b1.s##suffix |= (unsigned int)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + (b2.s##suffix >> 0) & 0xff)]; \
                b2.s##suffix &= 0xffffff00;\
                passOffset++;\
                b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  b2.s##suffix &= 0xffffff00;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + (b2.s##suffix >> 8) & 0xff)]; \
                  b2.s##suffix &= 0xffff00ff;\
                  passOffset++;\
                  b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    b2.s##suffix &= 0xffff00ff;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + (b2.s##suffix >> 16) & 0xff)]; \
                    b2.s##suffix &= 0xff00ffff;\
                    passOffset++;\
                    b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      b2.s##suffix &= 0xff00ffff;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + (b2.s##suffix >> 24) & 0xff)]; \
                      b2.s##suffix &= 0x00ffffff;\
                      passOffset++;\
                      b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        b2.s##suffix &= 0x00ffffff;\
                        b2.s##suffix |= (unsigned int)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + (b3.s##suffix >> 0) & 0xff)]; \
                        b3.s##suffix &= 0xffffff00;\
                        passOffset++;\
                        b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          b3.s##suffix &= 0xffffff00;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + (b3.s##suffix >> 8) & 0xff)]; \
                          b3.s##suffix &= 0xffff00ff;\
                          passOffset++;\
                          b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            b3.s##suffix &= 0xffff00ff;\
                            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 13)] << 8);\
                            passOffset = charsetReverse[((128 * 14) + (b3.s##suffix >> 16) & 0xff)]; \
                            b3.s##suffix &= 0xff00ffff;\
                            passOffset++;\
                            b3.s##suffix |= (unsigned int)(charsetForward[(128 * 14) + passOffset] << 16);\
                            if (passOffset >= charsetLengths[14]) { \
                              b3.s##suffix &= 0xff00ffff;\
                              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 14)] << 16);\
                              passOffset = charsetReverse[((128 * 15) + (b3.s##suffix >> 24) & 0xff)]; \
                              b3.s##suffix &= 0x00ffffff;\
                              passOffset++;\
                              b3.s##suffix |= (unsigned int)(charsetForward[(128 * 15) + passOffset] << 24);\
                              if (passOffset >= charsetLengths[15]) { \
                                b3.s##suffix &= 0x00ffffff;\
                                b3.s##suffix |= (unsigned int)(charsetForward[(128 * 15)] << 24);\
} } } } } } } } } } } } } } } } } 


