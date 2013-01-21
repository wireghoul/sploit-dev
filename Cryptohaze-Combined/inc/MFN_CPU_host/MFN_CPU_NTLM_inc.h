#define makeMFNSingleIncrementorsCPU_NTLM1(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
} } 


#define makeMFNSingleIncrementorsCPU_NTLM2(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
} } } 


#define makeMFNSingleIncrementorsCPU_NTLM3(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
} } } } 


#define makeMFNSingleIncrementorsCPU_NTLM4(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM5(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM6(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM7(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM8(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM9(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM10(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM11(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM12(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(w5 >> 16) & 0xff]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM13(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(w5 >> 16) & 0xff]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(w6 >> 0) & 0xff]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM14(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(w5 >> 16) & 0xff]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(w6 >> 0) & 0xff]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w6 >> 16) & 0xff]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM15(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(w5 >> 16) & 0xff]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(w6 >> 0) & 0xff]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w6 >> 16) & 0xff]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[0] << 16);\
                            passOffset = charsetReverse[(w7 >> 0) & 0xff]; \
                            w7 &= 0xffffff00;\
                            passOffset++;\
                            w7 |= (uint32_t)(charsetForward[passOffset] << 0);\
                            if (passOffset >= charsetLengths[0]) { \
                              w7 &= 0xffffff00;\
                              w7 |= (uint32_t)(charsetForward[0] << 0);\
} } } } } } } } } } } } } } } } 


#define makeMFNSingleIncrementorsCPU_NTLM16(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[(w0 >> 0) & 0xff]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[0] << 0);\
  passOffset = charsetReverse[(w0 >> 16) & 0xff]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[passOffset] << 16);\
  if (passOffset >= charsetLengths[0]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[0] << 16);\
    passOffset = charsetReverse[(w1 >> 0) & 0xff]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[passOffset] << 0);\
    if (passOffset >= charsetLengths[0]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[0] << 0);\
      passOffset = charsetReverse[(w1 >> 16) & 0xff]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[passOffset] << 16);\
      if (passOffset >= charsetLengths[0]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[0] << 16);\
        passOffset = charsetReverse[(w2 >> 0) & 0xff]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[passOffset] << 0);\
        if (passOffset >= charsetLengths[0]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[0] << 0);\
          passOffset = charsetReverse[(w2 >> 16) & 0xff]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[passOffset] << 16);\
          if (passOffset >= charsetLengths[0]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[0] << 16);\
            passOffset = charsetReverse[(w3 >> 0) & 0xff]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[passOffset] << 0);\
            if (passOffset >= charsetLengths[0]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[0] << 0);\
              passOffset = charsetReverse[(w3 >> 16) & 0xff]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[passOffset] << 16);\
              if (passOffset >= charsetLengths[0]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[0] << 16);\
                passOffset = charsetReverse[(w4 >> 0) & 0xff]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[passOffset] << 0);\
                if (passOffset >= charsetLengths[0]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[0] << 0);\
                  passOffset = charsetReverse[(w4 >> 16) & 0xff]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[passOffset] << 16);\
                  if (passOffset >= charsetLengths[0]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[0] << 16);\
                    passOffset = charsetReverse[(w5 >> 0) & 0xff]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[passOffset] << 0);\
                    if (passOffset >= charsetLengths[0]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[0] << 0);\
                      passOffset = charsetReverse[(w5 >> 16) & 0xff]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[passOffset] << 16);\
                      if (passOffset >= charsetLengths[0]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[0] << 16);\
                        passOffset = charsetReverse[(w6 >> 0) & 0xff]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[passOffset] << 0);\
                        if (passOffset >= charsetLengths[0]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[0] << 0);\
                          passOffset = charsetReverse[(w6 >> 16) & 0xff]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[passOffset] << 16);\
                          if (passOffset >= charsetLengths[0]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[0] << 16);\
                            passOffset = charsetReverse[(w7 >> 0) & 0xff]; \
                            w7 &= 0xffffff00;\
                            passOffset++;\
                            w7 |= (uint32_t)(charsetForward[passOffset] << 0);\
                            if (passOffset >= charsetLengths[0]) { \
                              w7 &= 0xffffff00;\
                              w7 |= (uint32_t)(charsetForward[0] << 0);\
                              passOffset = charsetReverse[(w7 >> 16) & 0xff]; \
                              w7 &= 0xff00ffff;\
                              passOffset++;\
                              w7 |= (uint32_t)(charsetForward[passOffset] << 16);\
                              if (passOffset >= charsetLengths[0]) { \
                                w7 &= 0xff00ffff;\
                                w7 |= (uint32_t)(charsetForward[0] << 16);\
} } } } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM1(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
} } 


#define makeMFNMultipleIncrementorsCPU_NTLM2(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
} } } 


#define makeMFNMultipleIncrementorsCPU_NTLM3(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
} } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM4(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
} } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM5(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
} } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM6(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
} } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM7(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
} } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM8(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
} } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM9(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
} } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM10(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
} } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM11(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
} } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM12(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + ((w5 >> 16) & 0xff))]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[(128 * 11)] << 16);\
} } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM13(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + ((w5 >> 16) & 0xff))]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + ((w6 >> 0) & 0xff))]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
} } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM14(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + ((w5 >> 16) & 0xff))]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + ((w6 >> 0) & 0xff))]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w6 >> 16) & 0xff))]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[(128 * 13)] << 16);\
} } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM15(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + ((w5 >> 16) & 0xff))]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + ((w6 >> 0) & 0xff))]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w6 >> 16) & 0xff))]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[(128 * 13)] << 16);\
                            passOffset = charsetReverse[((128 * 14) + ((w7 >> 0) & 0xff))]; \
                            w7 &= 0xffffff00;\
                            passOffset++;\
                            w7 |= (uint32_t)(charsetForward[(128 * 14) + passOffset] << 0);\
                            if (passOffset >= charsetLengths[14]) { \
                              w7 &= 0xffffff00;\
                              w7 |= (uint32_t)(charsetForward[(128 * 14)] << 0);\
} } } } } } } } } } } } } } } } 


#define makeMFNMultipleIncrementorsCPU_NTLM16(charsetForward, charsetReverse, charsetLengths) {\
passOffset = charsetReverse[((128 * 0) + ((w0 >> 0) & 0xff))]; \
w0 &= 0xffffff00;\
passOffset++;\
w0 |= (uint32_t)(charsetForward[(128 * 0) + passOffset] << 0);\
if (passOffset >= charsetLengths[0]) { \
  w0 &= 0xffffff00;\
  w0 |= (uint32_t)(charsetForward[(128 * 0)] << 0);\
  passOffset = charsetReverse[((128 * 1) + ((w0 >> 16) & 0xff))]; \
  w0 &= 0xff00ffff;\
  passOffset++;\
  w0 |= (uint32_t)(charsetForward[(128 * 1) + passOffset] << 16);\
  if (passOffset >= charsetLengths[1]) { \
    w0 &= 0xff00ffff;\
    w0 |= (uint32_t)(charsetForward[(128 * 1)] << 16);\
    passOffset = charsetReverse[((128 * 2) + ((w1 >> 0) & 0xff))]; \
    w1 &= 0xffffff00;\
    passOffset++;\
    w1 |= (uint32_t)(charsetForward[(128 * 2) + passOffset] << 0);\
    if (passOffset >= charsetLengths[2]) { \
      w1 &= 0xffffff00;\
      w1 |= (uint32_t)(charsetForward[(128 * 2)] << 0);\
      passOffset = charsetReverse[((128 * 3) + ((w1 >> 16) & 0xff))]; \
      w1 &= 0xff00ffff;\
      passOffset++;\
      w1 |= (uint32_t)(charsetForward[(128 * 3) + passOffset] << 16);\
      if (passOffset >= charsetLengths[3]) { \
        w1 &= 0xff00ffff;\
        w1 |= (uint32_t)(charsetForward[(128 * 3)] << 16);\
        passOffset = charsetReverse[((128 * 4) + ((w2 >> 0) & 0xff))]; \
        w2 &= 0xffffff00;\
        passOffset++;\
        w2 |= (uint32_t)(charsetForward[(128 * 4) + passOffset] << 0);\
        if (passOffset >= charsetLengths[4]) { \
          w2 &= 0xffffff00;\
          w2 |= (uint32_t)(charsetForward[(128 * 4)] << 0);\
          passOffset = charsetReverse[((128 * 5) + ((w2 >> 16) & 0xff))]; \
          w2 &= 0xff00ffff;\
          passOffset++;\
          w2 |= (uint32_t)(charsetForward[(128 * 5) + passOffset] << 16);\
          if (passOffset >= charsetLengths[5]) { \
            w2 &= 0xff00ffff;\
            w2 |= (uint32_t)(charsetForward[(128 * 5)] << 16);\
            passOffset = charsetReverse[((128 * 6) + ((w3 >> 0) & 0xff))]; \
            w3 &= 0xffffff00;\
            passOffset++;\
            w3 |= (uint32_t)(charsetForward[(128 * 6) + passOffset] << 0);\
            if (passOffset >= charsetLengths[6]) { \
              w3 &= 0xffffff00;\
              w3 |= (uint32_t)(charsetForward[(128 * 6)] << 0);\
              passOffset = charsetReverse[((128 * 7) + ((w3 >> 16) & 0xff))]; \
              w3 &= 0xff00ffff;\
              passOffset++;\
              w3 |= (uint32_t)(charsetForward[(128 * 7) + passOffset] << 16);\
              if (passOffset >= charsetLengths[7]) { \
                w3 &= 0xff00ffff;\
                w3 |= (uint32_t)(charsetForward[(128 * 7)] << 16);\
                passOffset = charsetReverse[((128 * 8) + ((w4 >> 0) & 0xff))]; \
                w4 &= 0xffffff00;\
                passOffset++;\
                w4 |= (uint32_t)(charsetForward[(128 * 8) + passOffset] << 0);\
                if (passOffset >= charsetLengths[8]) { \
                  w4 &= 0xffffff00;\
                  w4 |= (uint32_t)(charsetForward[(128 * 8)] << 0);\
                  passOffset = charsetReverse[((128 * 9) + ((w4 >> 16) & 0xff))]; \
                  w4 &= 0xff00ffff;\
                  passOffset++;\
                  w4 |= (uint32_t)(charsetForward[(128 * 9) + passOffset] << 16);\
                  if (passOffset >= charsetLengths[9]) { \
                    w4 &= 0xff00ffff;\
                    w4 |= (uint32_t)(charsetForward[(128 * 9)] << 16);\
                    passOffset = charsetReverse[((128 * 10) + ((w5 >> 0) & 0xff))]; \
                    w5 &= 0xffffff00;\
                    passOffset++;\
                    w5 |= (uint32_t)(charsetForward[(128 * 10) + passOffset] << 0);\
                    if (passOffset >= charsetLengths[10]) { \
                      w5 &= 0xffffff00;\
                      w5 |= (uint32_t)(charsetForward[(128 * 10)] << 0);\
                      passOffset = charsetReverse[((128 * 11) + ((w5 >> 16) & 0xff))]; \
                      w5 &= 0xff00ffff;\
                      passOffset++;\
                      w5 |= (uint32_t)(charsetForward[(128 * 11) + passOffset] << 16);\
                      if (passOffset >= charsetLengths[11]) { \
                        w5 &= 0xff00ffff;\
                        w5 |= (uint32_t)(charsetForward[(128 * 11)] << 16);\
                        passOffset = charsetReverse[((128 * 12) + ((w6 >> 0) & 0xff))]; \
                        w6 &= 0xffffff00;\
                        passOffset++;\
                        w6 |= (uint32_t)(charsetForward[(128 * 12) + passOffset] << 0);\
                        if (passOffset >= charsetLengths[12]) { \
                          w6 &= 0xffffff00;\
                          w6 |= (uint32_t)(charsetForward[(128 * 12)] << 0);\
                          passOffset = charsetReverse[((128 * 13) + ((w6 >> 16) & 0xff))]; \
                          w6 &= 0xff00ffff;\
                          passOffset++;\
                          w6 |= (uint32_t)(charsetForward[(128 * 13) + passOffset] << 16);\
                          if (passOffset >= charsetLengths[13]) { \
                            w6 &= 0xff00ffff;\
                            w6 |= (uint32_t)(charsetForward[(128 * 13)] << 16);\
                            passOffset = charsetReverse[((128 * 14) + ((w7 >> 0) & 0xff))]; \
                            w7 &= 0xffffff00;\
                            passOffset++;\
                            w7 |= (uint32_t)(charsetForward[(128 * 14) + passOffset] << 0);\
                            if (passOffset >= charsetLengths[14]) { \
                              w7 &= 0xffffff00;\
                              w7 |= (uint32_t)(charsetForward[(128 * 14)] << 0);\
                              passOffset = charsetReverse[((128 * 15) + ((w7 >> 16) & 0xff))]; \
                              w7 &= 0xff00ffff;\
                              passOffset++;\
                              w7 |= (uint32_t)(charsetForward[(128 * 15) + passOffset] << 16);\
                              if (passOffset >= charsetLengths[15]) { \
                                w7 &= 0xff00ffff;\
                                w7 |= (uint32_t)(charsetForward[(128 * 15)] << 16);\
} } } } } } } } } } } } } } } } } 


