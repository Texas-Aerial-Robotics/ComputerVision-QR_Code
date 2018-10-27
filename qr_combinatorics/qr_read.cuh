#ifndef QR_READ_H_
#define QR_READ_H_

__device__ int extract(unsigned char *bytes, int pos, int len){
	int shift = 24 - (pos & 7) - len;
	int mask = (1 << len) - 1;
	int byteIndex = pos >> 3;
	return (((bytes[byteIndex] << 16) | (bytes[++byteIndex] << 8) | bytes[++byteIndex]) >> shift) & mask;
}

struct digit4 {
	unsigned char a, b, c, d;
	bool good;
};

__device__ digit4 extract_numeric(unsigned char *bytes, int bit_idx){
	digit4 ret {0, 0, 0, 0, false};
	int n = extract(bytes, bit_idx, 10);
	if(n != 4) return ret;
	bit_idx += 10;
	int x = extract(bytes, bit_idx, 10);
	bit_idx += 10;
	ret.a = x / 100;
	if(ret.a >= 10) return ret;
	ret.b = (x % 100) / 10;
	if(ret.b >= 10) return ret;
	ret.c = x % 10;
	if(ret.c >= 10) return ret;
	x = extract(bytes, bit_idx, 4);
	bit_idx += 4;
	if(x >= 10) return ret;
	ret.d = x;
	ret.good = true;
	return ret;
}

#endif
