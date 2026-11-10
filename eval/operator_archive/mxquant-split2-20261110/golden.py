"""Frozen independent host BF16/MXFP8 scalar reference; Python standard library only."""
import struct
def round_even(v,s):
 if s<=0:return v<<(-s)
 q,rem=divmod(v,1<<s);half=1<<(s-1)
 return q+int(rem>half or (rem==half and q&1))
def e4(bits):
 sign=(bits>>8)&128;v=bits&32767;e=(v>>7)-127
 if v>>7==0:return sign
 if e>8:return sign|126
 if e<-9:return sign
 if e<-6:return sign|min(round_even(128+(v&127),-(e+2)),8)
 m=round_even(v&127,4);be=e+7
 if m>=8:m=0;be+=1
 return sign|(126 if be>15 or (be==15 and m>6) else (be<<3)|m)
def input_and_golden():
 vals=[];payload=bytearray();scales=bytearray();state=12345
 for group in range(4096):
  v=[];base=108+(group*7+group//8*3)%32
  for col in range(32):
   state=(state*1664525+1013904223)&0xffffffff
   bits=(((state>>8)&1)<<15)|((base+((state>>16)%8))<<7)|((state>>24)&127)
   if col==0 and group%5==0:bits&=32768
   v.append(bits)
  vals.extend(v);amax=max(x&32767 for x in v);code=(amax>>7)-8;scales.append(code)
  for bits in v:
   ab=bits&32767;exp=ab>>7
   scaled=(bits&32768)|((exp-(code-127))<<7)|(ab&127) if exp else bits
   payload.append(e4(scaled))
 return struct.pack('<'+str(len(vals))+'H',*vals),bytes(payload),bytes(scales)
def compare(a,b):
 if len(a)!=len(b):return {'pass':False,'actual_bytes':len(a),'expected_bytes':len(b)}
 bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
 return {'pass':not bad,'bytes':len(b),'mismatches':len(bad),'first':bad[:8]}
