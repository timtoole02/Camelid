
target/release/camelid:     file format elf64-x86-64


Disassembly of section .text:

0000000000366fd0 <camelid::inference::q8_0_packed_4x8_block_avx512vnni_dpwssd>:
  366fd0:	62 f2 7d 48 20 06    	vpmovsxbw (%rsi),%zmm0
  366fd6:	c4 e2 7d 59 0a       	vpbroadcastq (%rdx),%ymm1
  366fdb:	62 f2 7d 48 20 c9    	vpmovsxbw %ymm1,%zmm1
  366fe1:	c5 e9 ef d2          	vpxor  %xmm2,%xmm2,%xmm2
  366fe5:	62 f2 7d 48 20 5e 01 	vpmovsxbw 0x20(%rsi),%zmm3
  366fec:	62 f2 7d 48 52 d1    	vpdpwssd %zmm1,%zmm0,%zmm2
  366ff2:	c4 e2 7d 59 42 08    	vpbroadcastq 0x8(%rdx),%ymm0
  366ff8:	62 f2 7d 48 20 c0    	vpmovsxbw %ymm0,%zmm0
  366ffe:	62 f1 65 48 f5 c0    	vpmaddwd %zmm0,%zmm3,%zmm0
  367004:	62 f1 6d 48 fe c0    	vpaddd %zmm0,%zmm2,%zmm0
  36700a:	62 f2 7d 48 20 4e 02 	vpmovsxbw 0x40(%rsi),%zmm1
  367011:	c4 e2 7d 59 52 10    	vpbroadcastq 0x10(%rdx),%ymm2
  367017:	62 f2 7d 48 20 d2    	vpmovsxbw %ymm2,%zmm2
  36701d:	62 f1 75 48 f5 ca    	vpmaddwd %zmm2,%zmm1,%zmm1
  367023:	62 f2 7d 48 20 56 03 	vpmovsxbw 0x60(%rsi),%zmm2
  36702a:	c4 e2 7d 59 5a 18    	vpbroadcastq 0x18(%rdx),%ymm3
  367030:	62 f1 7d 48 fe c1    	vpaddd %zmm1,%zmm0,%zmm0
  367036:	62 f2 7d 48 20 cb    	vpmovsxbw %ymm3,%zmm1
  36703c:	62 f1 6d 48 f5 c9    	vpmaddwd %zmm1,%zmm2,%zmm1
  367042:	62 f1 7d 48 fe c1    	vpaddd %zmm1,%zmm0,%zmm0
  367048:	c5 f9 6f 0d 00 24 cd 	vmovdqa -0x32dc00(%rip),%xmm1        # 39450 <anon.15b9b9851f81874839a44a8fb676dc10.107.llvm.13004447194932167009+0x2f0>
  36704f:	ff
  367050:	62 f2 75 48 36 c8    	vpermd %zmm0,%zmm1,%zmm1
  367056:	c5 f9 6f 15 42 37 cd 	vmovdqa -0x32c8be(%rip),%xmm2        # 3a7a0 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.301.llvm.15554514708717879527+0x10>
  36705d:	ff
  36705e:	62 f2 6d 48 36 d0    	vpermd %zmm0,%zmm2,%zmm2
  367064:	c5 f9 6f 1d 24 2b cd 	vmovdqa -0x32d4dc(%rip),%xmm3        # 39b90 <anon.fae1a223151f84b017ca6fb5fafc86af.39.llvm.12260997531115035304+0xc0>
  36706b:	ff
  36706c:	62 f2 65 48 36 d8    	vpermd %zmm0,%zmm3,%zmm3
  367072:	c5 f9 6f 25 a6 34 cd 	vmovdqa -0x32cb5a(%rip),%xmm4        # 3a520 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.235.llvm.15554514708717879527+0x10>
  367079:	ff
  36707a:	62 f2 5d 48 36 c0    	vpermd %zmm0,%zmm4,%zmm0
  367080:	c5 f9 fe c3          	vpaddd %xmm3,%xmm0,%xmm0
  367084:	c5 f1 fe ca          	vpaddd %xmm2,%xmm1,%xmm1
  367088:	c5 f1 fe c0          	vpaddd %xmm0,%xmm1,%xmm0
  36708c:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  367090:	c5 f8 77             	vzeroupper
  367093:	c3                   	ret
  367094:	cc                   	int3
  367095:	cc                   	int3
  367096:	cc                   	int3
  367097:	cc                   	int3
  367098:	cc                   	int3
  367099:	cc                   	int3
  36709a:	cc                   	int3
  36709b:	cc                   	int3
  36709c:	cc                   	int3
  36709d:	cc                   	int3
  36709e:	cc                   	int3
  36709f:	cc                   	int3

00000000003670a0 <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into>:
  3670a0:	55                   	push   %rbp
  3670a1:	41 57                	push   %r15
  3670a3:	41 56                	push   %r14
  3670a5:	41 55                	push   %r13
  3670a7:	41 54                	push   %r12
  3670a9:	53                   	push   %rbx
  3670aa:	48 81 ec 48 01 00 00 	sub    $0x148,%rsp
  3670b1:	4c 89 c0             	mov    %r8,%rax
  3670b4:	49 89 cf             	mov    %rcx,%r15
  3670b7:	49 89 f6             	mov    %rsi,%r14
  3670ba:	49 89 d5             	mov    %rdx,%r13
  3670bd:	49 c1 ed 02          	shr    $0x2,%r13
  3670c1:	4c 89 44 24 48       	mov    %r8,0x48(%rsp)
  3670c6:	49 0f af c5          	imul   %r13,%rax
  3670ca:	49 8b 09             	mov    (%r9),%rcx
  3670cd:	49 8b 71 10          	mov    0x10(%r9),%rsi
  3670d1:	48 29 f1             	sub    %rsi,%rcx
  3670d4:	48 39 c8             	cmp    %rcx,%rax
  3670d7:	4c 89 4c 24 58       	mov    %r9,0x58(%rsp)
  3670dc:	0f 87 01 05 00 00    	ja     3675e3 <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into+0x543>
  3670e2:	48 85 d2             	test   %rdx,%rdx
  3670e5:	0f 84 23 05 00 00    	je     36760e <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into+0x56e>
  3670eb:	48 83 7c 24 48 00    	cmpq   $0x0,0x48(%rsp)
  3670f1:	0f 84 17 05 00 00    	je     36760e <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into+0x56e>
  3670f7:	83 e2 03             	and    $0x3,%edx
  3670fa:	48 83 fa 01          	cmp    $0x1,%rdx
  3670fe:	49 83 dd ff          	sbb    $0xffffffffffffffff,%r13
  367102:	4a 8d 04 bd 00 00 00 	lea    0x0(,%r15,4),%rax
  367109:	00
  36710a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  36710f:	48 8d 04 40          	lea    (%rax,%rax,2),%rax
  367113:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
  36711a:	00
  36711b:	4c 89 f8             	mov    %r15,%rax
  36711e:	48 c1 e0 04          	shl    $0x4,%rax
  367122:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
  367127:	4b 8d 04 7f          	lea    (%r15,%r15,2),%rax
  36712b:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  367132:	00
  367133:	4a 8d 04 fd 00 00 00 	lea    0x0(,%r15,8),%rax
  36713a:	00
  36713b:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  367142:	00
  367143:	4f 8d 24 3f          	lea    (%r15,%r15,1),%r12
  367147:	31 ed                	xor    %ebp,%ebp
  367149:	31 c0                	xor    %eax,%eax
  36714b:	4c 89 bc 24 c8 00 00 	mov    %r15,0xc8(%rsp)
  367152:	00
  367153:	4c 89 a4 24 b8 00 00 	mov    %r12,0xb8(%rsp)
  36715a:	00
  36715b:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
  367160:	eb 3e                	jmp    3671a0 <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into+0x100>
  367162:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  367169:	1f 84 00 00 00 00 00
  367170:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  367177:	00
  367178:	48 03 7c 24 60       	add    0x60(%rsp),%rdi
  36717d:	48 8b ac 24 80 00 00 	mov    0x80(%rsp),%rbp
  367184:	00
  367185:	48 03 6c 24 50       	add    0x50(%rsp),%rbp
  36718a:	4c 8b ac 24 88 00 00 	mov    0x88(%rsp),%r13
  367191:	00
  367192:	4d 85 ed             	test   %r13,%r13
  367195:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  36719a:	0f 84 6e 04 00 00    	je     36760e <camelid::inference::quantize_pack_q8_0_rows4_i8_direct_into+0x56e>
