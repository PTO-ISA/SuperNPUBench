#ifdef __cplusplus
extern "C" {
#endif

void *memset(void *dst, int value, unsigned long size) {
  volatile unsigned char *out = (volatile unsigned char *)dst;
  for (unsigned long i = 0; i < size; ++i)
    out[i] = (unsigned char)value;
  return dst;
}

void *memcpy(void *dst, const void *src, unsigned long size) {
  volatile unsigned char *out = (volatile unsigned char *)dst;
  const volatile unsigned char *in = (const volatile unsigned char *)src;
  for (unsigned long i = 0; i < size; ++i)
    out[i] = in[i];
  return dst;
}

void *memmove(void *dst, const void *src, unsigned long size) {
  volatile unsigned char *out = (volatile unsigned char *)dst;
  const volatile unsigned char *in = (const volatile unsigned char *)src;
  if (out < in) {
    for (unsigned long i = 0; i < size; ++i)
      out[i] = in[i];
  } else if (out > in) {
    for (unsigned long i = size; i != 0; --i)
      out[i - 1] = in[i - 1];
  }
  return dst;
}

#ifdef __cplusplus
}
#endif
