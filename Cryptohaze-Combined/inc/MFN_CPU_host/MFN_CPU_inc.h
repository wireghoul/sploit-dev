#define makeMFNSingleIncrementorsCPU1(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
} } 


#define makeMFNSingleIncrementorsCPU2(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
} } } 


#define makeMFNSingleIncrementorsCPU3(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
} } } } 


#define makeMFNSingleIncrementorsCPU4(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
} } } } } 


#define makeMFNSingleIncrementorsCPU5(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } 


#define makeMFNSingleIncrementorsCPU6(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
} } } } } } } 


#define makeMFNSingleIncrementorsCPU7(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } 


#define makeMFNSingleIncrementorsCPU8(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
} } } } } } } } } 


#define makeMFNSingleIncrementorsCPU9(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU10(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
} } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU11(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU12(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(w2 >> 24) & 0xff]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[0] << 24);\
} } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU13(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(w2 >> 24) & 0xff]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU14(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(w2 >> 24) & 0xff]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w3 >> 8) & 0xff]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[0] << 8);\
} } } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU15(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(w2 >> 24) & 0xff]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w3 >> 8) & 0xff]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[0] << 8);\
                            passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
                            w3 &= 0xff00ffff;\
                            passOffset++;\
                            w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
                            if (passOffset >= charsetLengths[0]) { \
                              w3 &= 0xff00ffff;\
                              w3 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU16(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 8) & 0xff]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 8);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[0] << 8);\
    passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
    if (passOffset >= charsetLengths[0]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[0] << 16);\
      passOffset = charsetReverse[(w0 >> 24) & 0xff]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[passOffset] << 24);\
      if (passOffset >= charsetLengths[0]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[0] << 24);\
        passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w1 >> 8) & 0xff]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[passOffset] << 8);\
          if (passOffset >= charsetLengths[0]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[0] << 8);\
            passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
            if (passOffset >= charsetLengths[0]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[0] << 16);\
              passOffset = charsetReverse[(w1 >> 24) & 0xff]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[passOffset] << 24);\
              if (passOffset >= charsetLengths[0]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[0] << 24);\
                passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w2 >> 8) & 0xff]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[passOffset] << 8);\
                  if (passOffset >= charsetLengths[0]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[0] << 8);\
                    passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
                    if (passOffset >= charsetLengths[0]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[0] << 16);\
                      passOffset = charsetReverse[(w2 >> 24) & 0xff]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[passOffset] << 24);\
                      if (passOffset >= charsetLengths[0]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[0] << 24);\
                        passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w3 >> 8) & 0xff]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[passOffset] << 8);\
                          if (passOffset >= charsetLengths[0]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[0] << 8);\
                            passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
                            w3 &= 0xff00ffff;\
                            passOffset++;\
                            w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
                            if (passOffset >= charsetLengths[0]) { \
                              w3 &= 0xff00ffff;\
                              w3 |= (uint32_t)(charsetForward[0] << 16);\
                              passOffset = charsetReverse[(w3 >> 24) & 0xff]; \
                              w3 &= 0x00ffffff;\
                              passOffset++;\
                              w3 |= (uint32_t)(charsetForward[passOffset] << 24);\
                              if (passOffset >= charsetLengths[0]) { \
                                w3 &= 0x00ffffff;\
                                w3 |= (uint32_t)(charsetForward[0] << 24);\
} } } } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU1(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
} } 


#define makeMFNMultipleIncrementorsCPU2(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
} } } 


#define makeMFNMultipleIncrementorsCPU3(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
} } } } 


#define makeMFNMultipleIncrementorsCPU4(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
} } } } } 


#define makeMFNMultipleIncrementorsCPU5(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
} } } } } } 


#define makeMFNMultipleIncrementorsCPU6(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
} } } } } } } 


#define makeMFNMultipleIncrementorsCPU7(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
} } } } } } } } 


#define makeMFNMultipleIncrementorsCPU8(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
} } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU9(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
} } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU10(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
} } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU11(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
} } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU12(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + ((w2 >> 24) & 0xff))]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[(128 * 11)] << 24);\
} } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU13(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + ((w2 >> 24) & 0xff))]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + ((w3 >> 0) & 0xff))]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
} } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU14(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + ((w2 >> 24) & 0xff))]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + ((w3 >> 0) & 0xff))]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w3 >> 8) & 0xff))]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[(128 * 13)] << 8);\
} } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU15(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + ((w2 >> 24) & 0xff))]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + ((w3 >> 0) & 0xff))]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w3 >> 8) & 0xff))]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[(128 * 13)] << 8);\
                            passOffset = charsetReverse[((128 * 14) + ((w3 >> 16) & 0xff))]; \
                            w3 &= 0xff00ffff;\
                            passOffset++;\
                            w3 |= (uint32_t)(charsetForward[(128 * 14) + passOffset] << 16);\
                            if (passOffset >= charsetLengths[14]) { \
                              w3 &= 0xff00ffff;\
                              w3 |= (uint32_t)(charsetForward[(128 * 14)] << 16);\
} } } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU16(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 8) & 0xff))]; \
  w0 &= 0xffff00ff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 8);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xffff00ff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 8);\
    passOffset = charsetReverse[((128 * 2) + ((w0 >> 16) & 0xff))]; \
    w0 &= 0xff00ffff;\
    passOffset++;\
    w0 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 16);\
    if (passOffset >= charsetLengths[2]) { \
      w0 &= 0xff00ffff;\
      w0 |= (uint32_t)(charsetForward[(128 * 2)] << 16);\
      passOffset = charsetReverse[((128 * 3) + ((w0 >> 24) & 0xff))]; \
      w0 &= 0x00ffffff;\
      passOffset++;\
      w0 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 24);\
      if (passOffset >= charsetLengths[3]) { \
        w0 &= 0x00ffffff;\
        w0 |= (uint32_t)(charsetForward[(128 * 3)] << 24);\
        passOffset = charsetReverse[((128 * 4) + ((w1 >> 0) & 0xff))]; \
        w1 &= 0xffffff00;\
        passOffset++;\
        w1 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w1 &= 0xffffff00;\
          w1 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w1 >> 8) & 0xff))]; \
          w1 &= 0xffff00ff;\
          passOffset++;\
          w1 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 8);\
          if (passOffset >= charsetLengths[5]) { \
            w1 &= 0xffff00ff;\
            w1 |= (uint32_t)(charsetForward[(128 * 5)] << 8);\
            passOffset = charsetReverse[((128 * 6) + ((w1 >> 16) & 0xff))]; \
            w1 &= 0xff00ffff;\
            passOffset++;\
            w1 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 16);\
            if (passOffset >= charsetLengths[6]) { \
              w1 &= 0xff00ffff;\
              w1 |= (uint32_t)(charsetForward[(128 * 6)] << 16);\
              passOffset = charsetReverse[((128 * 7) + ((w1 >> 24) & 0xff))]; \
              w1 &= 0x00ffffff;\
              passOffset++;\
              w1 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 24);\
              if (passOffset >= charsetLengths[7]) { \
                w1 &= 0x00ffffff;\
                w1 |= (uint32_t)(charsetForward[(128 * 7)] << 24);\
                passOffset = charsetReverse[((128 * 8) + ((w2 >> 0) & 0xff))]; \
                w2 &= 0xffffff00;\
                passOffset++;\
                w2 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w2 &= 0xffffff00;\
                  w2 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w2 >> 8) & 0xff))]; \
                  w2 &= 0xffff00ff;\
                  passOffset++;\
                  w2 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 8);\
                  if (passOffset >= charsetLengths[9]) { \
                    w2 &= 0xffff00ff;\
                    w2 |= (uint32_t)(charsetForward[(128 * 9)] << 8);\
                    passOffset = charsetReverse[((128 * 10) + ((w2 >> 16) & 0xff))]; \
                    w2 &= 0xff00ffff;\
                    passOffset++;\
                    w2 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 16);\
                    if (passOffset >= charsetLengths[10]) { \
                      w2 &= 0xff00ffff;\
                      w2 |= (uint32_t)(charsetForward[(128 * 10)] << 16);\
                      passOffset = charsetReverse[((128 * 11) + ((w2 >> 24) & 0xff))]; \
                      w2 &= 0x00ffffff;\
                      passOffset++;\
                      w2 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 24);\
                      if (passOffset >= charsetLengths[11]) { \
                        w2 &= 0x00ffffff;\
                        w2 |= (uint32_t)(charsetForward[(128 * 11)] << 24);\
                        passOffset = charsetReverse[((128 * 12) + ((w3 >> 0) & 0xff))]; \
                        w3 &= 0xffffff00;\
                        passOffset++;\
                        w3 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w3 &= 0xffffff00;\
                          w3 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w3 >> 8) & 0xff))]; \
                          w3 &= 0xffff00ff;\
                          passOffset++;\
                          w3 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 8);\
                          if (passOffset >= charsetLengths[13]) { \
                            w3 &= 0xffff00ff;\
                            w3 |= (uint32_t)(charsetForward[(128 * 13)] << 8);\
                            passOffset = charsetReverse[((128 * 14) + ((w3 >> 16) & 0xff))]; \
                            w3 &= 0xff00ffff;\
                            passOffset++;\
                            w3 |= (uint32_t)(charsetForward[(128 * 14) + passOffset] << 16);\
                            if (passOffset >= charsetLengths[14]) { \
                              w3 &= 0xff00ffff;\
                              w3 |= (uint32_t)(charsetForward[(128 * 14)] << 16);\
                              passOffset = charsetReverse[((128 * 15) + ((w3 >> 24) & 0xff))]; \
                              w3 &= 0x00ffffff;\
                              passOffset++;\
                              w3 |= (uint32_t)(charsetForward[(128 * 15) + passOffset] << 24);\
                              if (passOffset >= charsetLengths[15]) { \
                                w3 &= 0x00ffffff;\
                                w3 |= (uint32_t)(charsetForward[(128 * 15)] << 24);\
} } } } } } } } } } } } } } } } } 


