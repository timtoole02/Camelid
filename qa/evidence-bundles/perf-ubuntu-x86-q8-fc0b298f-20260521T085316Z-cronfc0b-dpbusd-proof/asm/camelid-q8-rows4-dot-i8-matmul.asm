
target/release/camelid:     file format elf64-x86-64


Disassembly of section .text:

0000000000360950 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul>:
  360950:	55                   	push   %rbp
  360951:	41 57                	push   %r15
  360953:	41 56                	push   %r14
  360955:	41 55                	push   %r13
  360957:	41 54                	push   %r12
  360959:	53                   	push   %rbx
  36095a:	50                   	push   %rax
  36095b:	8b 05 77 40 33 00    	mov    0x334077(%rip),%eax        # 6949d8 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED>
  360961:	85 c0                	test   %eax,%eax
  360963:	0f 85 a6 02 00 00    	jne    360c0f <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x2bf>
  360969:	80 3d 6c 40 33 00 00 	cmpb   $0x0,0x33406c(%rip)        # 6949dc <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED+0x4>
  360970:	0f 84 d9 02 00 00    	je     360c4f <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x2ff>
  360976:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  36097a:	c5 f8 11 07          	vmovups %xmm0,(%rdi)
  36097e:	49 39 d0             	cmp    %rdx,%r8
  360981:	49 0f 42 d0          	cmovb  %r8,%rdx
  360985:	48 85 d2             	test   %rdx,%rdx
  360988:	0f 84 3e 05 00 00    	je     360ecc <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x57c>
  36098e:	48 83 fa 01          	cmp    $0x1,%rdx
  360992:	75 07                	jne    36099b <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x4b>
  360994:	31 c0                	xor    %eax,%eax
  360996:	e9 ac 01 00 00       	jmp    360b47 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x1f7>
  36099b:	49 b8 fe ff ff ff ff 	movabs $0xfffffffffffffe,%r8
  3609a2:	ff ff 00 
  3609a5:	49 21 d0             	and    %rdx,%r8
  3609a8:	41 b9 38 00 00 00    	mov    $0x38,%r9d
  3609ae:	31 c0                	xor    %eax,%eax
  3609b0:	62 f1 fd 48 6f 0d c6 	vmovdqa64 -0x301b3a(%rip),%zmm1        # 5ee80 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.432.llvm.9595541323801984552+0x56da>
  3609b7:	e4 cf ff 
  3609ba:	c5 e9 ef d2          	vpxor  %xmm2,%xmm2,%xmm2
  3609be:	c5 fd 6f 1d 9a b2 cd 	vmovdqa -0x324d66(%rip),%ymm3        # 3bc60 <anon.273cc08786297daf0067aca266fa7cbc.5.llvm.17933334302312970005+0x40>
  3609c5:	ff 
  3609c6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  3609cd:	00 00 00 
  3609d0:	62 b1 fe 48 6f a4 8e 	vmovdqu64 -0xd0(%rsi,%r9,4),%zmm4
  3609d7:	30 ff ff ff 
  3609db:	62 b1 fe 48 6f ac 8e 	vmovdqu64 -0x90(%rsi,%r9,4),%zmm5
  3609e2:	70 ff ff ff 
  3609e6:	62 b1 fe 48 6f 74 8e 	vmovdqu64 -0x40(%rsi,%r9,4),%zmm6
  3609ed:	ff 
  3609ee:	62 b1 fe 48 6f 3c 8e 	vmovdqu64 (%rsi,%r9,4),%zmm7
  3609f5:	62 72 7d 48 1c c4    	vpabsb %zmm4,%zmm8
  3609fb:	c4 21 7a 6f 4c 09 cc 	vmovdqu -0x34(%rcx,%r9,1),%xmm9
  360a02:	c4 21 7a 6f 54 09 dc 	vmovdqu -0x24(%rcx,%r9,1),%xmm10
  360a09:	c4 21 7a 6f 5c 09 f0 	vmovdqu -0x10(%rcx,%r9,1),%xmm11
  360a10:	c4 21 7a 6f 24 09    	vmovdqu (%rcx,%r9,1),%xmm12
  360a16:	62 52 f5 48 36 c9    	vpermq %zmm9,%zmm1,%zmm9
  360a1c:	62 f2 7e 48 29 cc    	vpmovb2m %zmm4,%k1
  360a22:	62 51 6d 49 f8 c9    	vpsubb %zmm9,%zmm2,%zmm9{%k1}
  360a28:	c5 d9 ef e4          	vpxor  %xmm4,%xmm4,%xmm4
  360a2c:	62 d2 3d 48 50 e1    	vpdpbusd %zmm9,%zmm8,%zmm4
  360a32:	62 72 7d 48 1c c5    	vpabsb %zmm5,%zmm8
  360a38:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  360a3d:	62 52 f5 48 36 ca    	vpermq %zmm10,%zmm1,%zmm9
  360a43:	62 f2 7e 48 29 cd    	vpmovb2m %zmm5,%k1
  360a49:	62 51 6d 49 f8 c9    	vpsubb %zmm9,%zmm2,%zmm9{%k1}
  360a4f:	62 d2 3d 48 50 e1    	vpdpbusd %zmm9,%zmm8,%zmm4
  360a55:	62 f2 7e 28 35 e5    	vpmovqd %ymm4,%xmm5
  360a5b:	c4 c3 7d 39 e0 01    	vextracti128 $0x1,%ymm4,%xmm8
  360a61:	c4 41 58 c6 c0 dd    	vshufps $0xdd,%xmm8,%xmm4,%xmm8
  360a67:	c5 b9 fe ed          	vpaddd %xmm5,%xmm8,%xmm5
  360a6b:	62 f3 fd 48 3b e4 01 	vextracti64x4 $0x1,%zmm4,%ymm4
  360a72:	62 d2 7e 28 35 e0    	vpmovqd %ymm4,%xmm8
  360a78:	c5 b9 fe ed          	vpaddd %xmm5,%xmm8,%xmm5
  360a7c:	c4 e2 65 36 e4       	vpermd %ymm4,%ymm3,%ymm4
  360a81:	c5 d1 fe e4          	vpaddd %xmm4,%xmm5,%xmm4
  360a85:	c5 f8 5b e4          	vcvtdq2ps %xmm4,%xmm4
  360a89:	c4 a1 58 59 a4 8e 20 	vmulps -0xe0(%rsi,%r9,4),%xmm4,%xmm4
  360a90:	ff ff ff 
  360a93:	62 b1 5c 18 59 64 09 	vmulps -0x38(%rcx,%r9,1){1to4},%xmm4,%xmm4
  360a9a:	f2 
  360a9b:	c5 f8 58 c4          	vaddps %xmm4,%xmm0,%xmm0
  360a9f:	62 f2 7d 48 1c e6    	vpabsb %zmm6,%zmm4
  360aa5:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  360aa9:	62 d2 f5 48 36 eb    	vpermq %zmm11,%zmm1,%zmm5
  360aaf:	62 f2 7e 48 29 ce    	vpmovb2m %zmm6,%k1
  360ab5:	62 f1 6d 49 f8 ed    	vpsubb %zmm5,%zmm2,%zmm5{%k1}
  360abb:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  360abf:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  360ac5:	62 f2 7d 48 1c e7    	vpabsb %zmm7,%zmm4
  360acb:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
  360acf:	62 d2 f5 48 36 ec    	vpermq %zmm12,%zmm1,%zmm5
  360ad5:	62 f2 7e 48 29 cf    	vpmovb2m %zmm7,%k1
  360adb:	62 f1 6d 49 f8 ed    	vpsubb %zmm5,%zmm2,%zmm5{%k1}
  360ae1:	62 f2 5d 48 50 f5    	vpdpbusd %zmm5,%zmm4,%zmm6
  360ae7:	62 f2 7e 28 35 f4    	vpmovqd %ymm6,%xmm4
  360aed:	c4 e3 7d 39 f5 01    	vextracti128 $0x1,%ymm6,%xmm5
  360af3:	c5 c8 c6 ed dd       	vshufps $0xdd,%xmm5,%xmm6,%xmm5
  360af8:	c5 d9 fe e5          	vpaddd %xmm5,%xmm4,%xmm4
  360afc:	62 f3 fd 48 3b f5 01 	vextracti64x4 $0x1,%zmm6,%ymm5
  360b03:	62 f2 7e 28 35 ee    	vpmovqd %ymm5,%xmm6
  360b09:	c5 d9 fe e6          	vpaddd %xmm6,%xmm4,%xmm4
  360b0d:	c4 e2 65 36 ed       	vpermd %ymm5,%ymm3,%ymm5
  360b12:	c5 d9 fe e5          	vpaddd %xmm5,%xmm4,%xmm4
  360b16:	c5 f8 5b e4          	vcvtdq2ps %xmm4,%xmm4
  360b1a:	c4 a1 58 59 64 8e b0 	vmulps -0x50(%rsi,%r9,4),%xmm4,%xmm4
  360b21:	62 b1 5c 18 59 64 09 	vmulps -0x14(%rcx,%r9,1){1to4},%xmm4,%xmm4
  360b28:	fb 
  360b29:	c5 f8 58 c4          	vaddps %xmm4,%xmm0,%xmm0
  360b2d:	48 83 c0 02          	add    $0x2,%rax
  360b31:	49 83 c1 48          	add    $0x48,%r9
  360b35:	49 39 c0             	cmp    %rax,%r8
  360b38:	0f 85 92 fe ff ff    	jne    3609d0 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x80>
  360b3e:	f6 c2 01             	test   $0x1,%dl
  360b41:	0f 84 81 03 00 00    	je     360ec8 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x578>
  360b47:	48 8d 04 c0          	lea    (%rax,%rax,8),%rax
  360b4b:	48 89 c2             	mov    %rax,%rdx
  360b4e:	48 c1 e2 04          	shl    $0x4,%rdx
  360b52:	62 f1 fe 48 6f 8c 16 	vmovdqu64 0x10(%rsi,%rdx,1),%zmm1
  360b59:	10 00 00 00 
  360b5d:	62 f1 fe 48 6f 94 16 	vmovdqu64 0x50(%rsi,%rdx,1),%zmm2
  360b64:	50 00 00 00 
  360b68:	62 f2 7d 48 1c d9    	vpabsb %zmm1,%zmm3
  360b6e:	c5 fa 6f 64 81 04    	vmovdqu 0x4(%rcx,%rax,4),%xmm4
  360b74:	c5 fa 6f 6c 81 14    	vmovdqu 0x14(%rcx,%rax,4),%xmm5
  360b7a:	62 f1 fd 48 6f 35 fc 	vmovdqa64 -0x301d04(%rip),%zmm6        # 5ee80 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.432.llvm.9595541323801984552+0x56da>
  360b81:	e2 cf ff 
  360b84:	62 f2 cd 48 36 e4    	vpermq %zmm4,%zmm6,%zmm4
  360b8a:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  360b8e:	62 f2 7e 48 29 c9    	vpmovb2m %zmm1,%k1
  360b94:	62 f1 45 49 f8 e4    	vpsubb %zmm4,%zmm7,%zmm4{%k1}
  360b9a:	62 f2 7d 48 1c ca    	vpabsb %zmm2,%zmm1
  360ba0:	62 f2 cd 48 36 ed    	vpermq %zmm5,%zmm6,%zmm5
  360ba6:	62 f2 7e 48 29 ca    	vpmovb2m %zmm2,%k1
  360bac:	62 f1 45 49 f8 ed    	vpsubb %zmm5,%zmm7,%zmm5{%k1}
  360bb2:	62 f2 65 48 50 fc    	vpdpbusd %zmm4,%zmm3,%zmm7
  360bb8:	62 f2 75 48 50 fd    	vpdpbusd %zmm5,%zmm1,%zmm7
  360bbe:	62 f2 7e 28 35 f9    	vpmovqd %ymm7,%xmm1
  360bc4:	c4 e3 7d 39 fa 01    	vextracti128 $0x1,%ymm7,%xmm2
  360bca:	c5 c0 c6 d2 dd       	vshufps $0xdd,%xmm2,%xmm7,%xmm2
  360bcf:	c5 f1 fe ca          	vpaddd %xmm2,%xmm1,%xmm1
  360bd3:	62 f3 fd 48 3b fa 01 	vextracti64x4 $0x1,%zmm7,%ymm2
  360bda:	62 f2 7e 28 35 d3    	vpmovqd %ymm2,%xmm3
  360be0:	c5 f1 fe cb          	vpaddd %xmm3,%xmm1,%xmm1
  360be4:	c4 e2 7d 5a 1d 43 8d 	vbroadcasti128 -0x3272bd(%rip),%ymm3        # 39930 <anon.5d13a23a69ccae08cbcca912f731f329.37.llvm.17069624656327407388+0x130>
  360beb:	cd ff 
  360bed:	c4 e2 65 36 d2       	vpermd %ymm2,%ymm3,%ymm2
  360bf2:	c5 f1 fe ca          	vpaddd %xmm2,%xmm1,%xmm1
  360bf6:	c5 f8 5b c9          	vcvtdq2ps %xmm1,%xmm1
  360bfa:	c5 f0 59 0c 16       	vmulps (%rsi,%rdx,1),%xmm1,%xmm1
  360bff:	62 f1 74 18 59 0c 81 	vmulps (%rcx,%rax,4){1to4},%xmm1,%xmm1
  360c06:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  360c0a:	e9 b9 02 00 00       	jmp    360ec8 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x578>
  360c0f:	48 8d 05 c2 3d 33 00 	lea    0x333dc2(%rip),%rax        # 6949d8 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED>
  360c16:	48 89 fb             	mov    %rdi,%rbx
  360c19:	48 89 c7             	mov    %rax,%rdi
  360c1c:	49 89 cf             	mov    %rcx,%r15
  360c1f:	49 89 f4             	mov    %rsi,%r12
  360c22:	49 89 d6             	mov    %rdx,%r14
  360c25:	4d 89 c5             	mov    %r8,%r13
  360c28:	44 89 cd             	mov    %r9d,%ebp
  360c2b:	e8 1a e9 12 00       	call   48f54a <std::sync::once_lock::OnceLock<T>::initialize>
  360c30:	41 89 e9             	mov    %ebp,%r9d
  360c33:	4d 89 e8             	mov    %r13,%r8
  360c36:	4c 89 f2             	mov    %r14,%rdx
  360c39:	4c 89 e6             	mov    %r12,%rsi
  360c3c:	4c 89 f9             	mov    %r15,%rcx
  360c3f:	48 89 df             	mov    %rbx,%rdi
  360c42:	80 3d 93 3d 33 00 00 	cmpb   $0x0,0x333d93(%rip)        # 6949dc <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpbusd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT_ENABLED+0x4>
  360c49:	0f 85 27 fd ff ff    	jne    360976 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x26>
  360c4f:	8b 05 8b 3d 33 00    	mov    0x333d8b(%rip),%eax        # 6949e0 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED>
  360c55:	85 c0                	test   %eax,%eax
  360c57:	0f 85 38 01 00 00    	jne    360d95 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x445>
  360c5d:	80 3d 80 3d 33 00 00 	cmpb   $0x0,0x333d80(%rip)        # 6949e4 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED+0x4>
  360c64:	0f 84 6b 01 00 00    	je     360dd5 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x485>
  360c6a:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  360c6e:	c5 f8 11 07          	vmovups %xmm0,(%rdi)
  360c72:	49 39 d0             	cmp    %rdx,%r8
  360c75:	49 0f 42 d0          	cmovb  %r8,%rdx
  360c79:	48 85 d2             	test   %rdx,%rdx
  360c7c:	0f 84 4a 02 00 00    	je     360ecc <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x57c>
  360c82:	48 c1 e2 02          	shl    $0x2,%rdx
  360c86:	48 8d 04 d2          	lea    (%rdx,%rdx,8),%rax
  360c8a:	31 d2                	xor    %edx,%edx
  360c8c:	c5 f9 6f 0d 8c 88 cd 	vmovdqa -0x327774(%rip),%xmm1        # 39520 <anon.15b9b9851f81874839a44a8fb676dc10.107.llvm.18319441687235125973+0x290>
  360c93:	ff 
  360c94:	c5 f9 6f 15 54 9a cd 	vmovdqa -0x3265ac(%rip),%xmm2        # 3a6f0 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.301.llvm.9595541323801984552+0x10>
  360c9b:	ff 
  360c9c:	c5 f9 6f 1d 1c 8f cd 	vmovdqa -0x3270e4(%rip),%xmm3        # 39bc0 <anon.fae1a223151f84b017ca6fb5fafc86af.39.llvm.4994568352209428481+0xa0>
  360ca3:	ff 
  360ca4:	c5 f9 6f 25 04 98 cd 	vmovdqa -0x3267fc(%rip),%xmm4        # 3a4b0 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.235.llvm.9595541323801984552+0x10>
  360cab:	ff 
  360cac:	0f 1f 40 00          	nopl   0x0(%rax)
  360cb0:	62 f2 7d 48 20 ac 96 	vpmovsxbw 0x10(%rsi,%rdx,4),%zmm5
  360cb7:	10 00 00 00 
  360cbb:	c4 e2 7d 59 74 11 04 	vpbroadcastq 0x4(%rcx,%rdx,1),%ymm6
  360cc2:	62 f2 7d 48 20 f6    	vpmovsxbw %ymm6,%zmm6
  360cc8:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  360ccc:	62 f2 55 48 52 fe    	vpdpwssd %zmm6,%zmm5,%zmm7
  360cd2:	62 f2 7d 48 20 ac 96 	vpmovsxbw 0x30(%rsi,%rdx,4),%zmm5
  360cd9:	30 00 00 00 
  360cdd:	c4 e2 7d 59 74 11 0c 	vpbroadcastq 0xc(%rcx,%rdx,1),%ymm6
  360ce4:	62 f2 7d 48 20 f6    	vpmovsxbw %ymm6,%zmm6
  360cea:	62 f1 55 48 f5 ee    	vpmaddwd %zmm6,%zmm5,%zmm5
  360cf0:	62 f1 45 48 fe ed    	vpaddd %zmm5,%zmm7,%zmm5
  360cf6:	62 f2 7d 48 20 b4 96 	vpmovsxbw 0x50(%rsi,%rdx,4),%zmm6
  360cfd:	50 00 00 00 
  360d01:	c4 e2 7d 59 7c 11 14 	vpbroadcastq 0x14(%rcx,%rdx,1),%ymm7
  360d08:	62 f2 7d 48 20 ff    	vpmovsxbw %ymm7,%zmm7
  360d0e:	62 f1 4d 48 f5 f7    	vpmaddwd %zmm7,%zmm6,%zmm6
  360d14:	62 f1 55 48 fe ee    	vpaddd %zmm6,%zmm5,%zmm5
  360d1a:	62 f2 7d 48 20 b4 96 	vpmovsxbw 0x70(%rsi,%rdx,4),%zmm6
  360d21:	70 00 00 00 
  360d25:	c4 e2 7d 59 7c 11 1c 	vpbroadcastq 0x1c(%rcx,%rdx,1),%ymm7
  360d2c:	62 f2 7d 48 20 ff    	vpmovsxbw %ymm7,%zmm7
  360d32:	62 f1 4d 48 f5 f7    	vpmaddwd %zmm7,%zmm6,%zmm6
  360d38:	62 f1 55 48 fe ee    	vpaddd %zmm6,%zmm5,%zmm5
  360d3e:	c5 c9 ef f6          	vpxor  %xmm6,%xmm6,%xmm6
  360d42:	62 f2 75 48 36 f5    	vpermd %zmm5,%zmm1,%zmm6
  360d48:	c5 c1 ef ff          	vpxor  %xmm7,%xmm7,%xmm7
  360d4c:	62 f2 6d 48 36 fd    	vpermd %zmm5,%zmm2,%zmm7
  360d52:	c4 41 39 ef c0       	vpxor  %xmm8,%xmm8,%xmm8
  360d57:	62 72 65 48 36 c5    	vpermd %zmm5,%zmm3,%zmm8
  360d5d:	62 f2 5d 48 36 ed    	vpermd %zmm5,%zmm4,%zmm5
  360d63:	c5 b9 fe ed          	vpaddd %xmm5,%xmm8,%xmm5
  360d67:	c5 c9 fe f7          	vpaddd %xmm7,%xmm6,%xmm6
  360d6b:	c5 c9 fe ed          	vpaddd %xmm5,%xmm6,%xmm5
  360d6f:	c5 f8 5b ed          	vcvtdq2ps %xmm5,%xmm5
  360d73:	c5 d0 59 2c 96       	vmulps (%rsi,%rdx,4),%xmm5,%xmm5
  360d78:	62 f1 54 18 59 2c 11 	vmulps (%rcx,%rdx,1){1to4},%xmm5,%xmm5
  360d7f:	c5 f8 58 c5          	vaddps %xmm5,%xmm0,%xmm0
  360d83:	48 83 c2 24          	add    $0x24,%rdx
  360d87:	48 39 d0             	cmp    %rdx,%rax
  360d8a:	0f 85 20 ff ff ff    	jne    360cb0 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x360>
  360d90:	e9 33 01 00 00       	jmp    360ec8 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x578>
  360d95:	48 8d 05 44 3c 33 00 	lea    0x333c44(%rip),%rax        # 6949e0 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED>
  360d9c:	48 89 fb             	mov    %rdi,%rbx
  360d9f:	48 89 c7             	mov    %rax,%rdi
  360da2:	49 89 cf             	mov    %rcx,%r15
  360da5:	49 89 f4             	mov    %rsi,%r12
  360da8:	49 89 d6             	mov    %rdx,%r14
  360dab:	4d 89 c5             	mov    %r8,%r13
  360dae:	44 89 cd             	mov    %r9d,%ebp
  360db1:	e8 50 e7 12 00       	call   48f506 <std::sync::once_lock::OnceLock<T>::initialize>
  360db6:	41 89 e9             	mov    %ebp,%r9d
  360db9:	4d 89 e8             	mov    %r13,%r8
  360dbc:	4c 89 f2             	mov    %r14,%rdx
  360dbf:	4c 89 e6             	mov    %r12,%rsi
  360dc2:	4c 89 f9             	mov    %r15,%rcx
  360dc5:	48 89 df             	mov    %rbx,%rdi
  360dc8:	80 3d 15 3c 33 00 00 	cmpb   $0x0,0x333c15(%rip)        # 6949e4 <camelid::inference::x86_q8_packed_rows4_avx512vnni_dpwssd_dot_enabled::X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT_ENABLED+0x4>
  360dcf:	0f 85 95 fe ff ff    	jne    360c6a <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x31a>
  360dd5:	45 84 c9             	test   %r9b,%r9b
  360dd8:	0f 84 00 01 00 00    	je     360ede <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x58e>
  360dde:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  360de2:	c5 f8 11 07          	vmovups %xmm0,(%rdi)
  360de6:	49 39 d0             	cmp    %rdx,%r8
  360de9:	49 0f 42 d0          	cmovb  %r8,%rdx
  360ded:	48 85 d2             	test   %rdx,%rdx
  360df0:	0f 84 d6 00 00 00    	je     360ecc <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x57c>
  360df6:	48 c1 e2 02          	shl    $0x2,%rdx
  360dfa:	48 8d 04 d2          	lea    (%rdx,%rdx,8),%rax
  360dfe:	31 d2                	xor    %edx,%edx
  360e00:	c4 e2 7d 58 0d 4f 9b 	vpbroadcastd -0x3264b1(%rip),%ymm1        # 3a958 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.215.llvm.9595541323801984552+0x30>
  360e07:	cd ff 
  360e09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  360e10:	c5 fe 6f 54 96 10    	vmovdqu 0x10(%rsi,%rdx,4),%ymm2
  360e16:	c5 fe 6f 5c 96 30    	vmovdqu 0x30(%rsi,%rdx,4),%ymm3
  360e1c:	c5 fe 6f 64 96 50    	vmovdqu 0x50(%rsi,%rdx,4),%ymm4
  360e22:	c5 fe 6f 6c 96 70    	vmovdqu 0x70(%rsi,%rdx,4),%ymm5
  360e28:	c4 e2 6d 08 f2       	vpsignb %ymm2,%ymm2,%ymm6
  360e2d:	c4 e2 7d 59 7c 11 04 	vpbroadcastq 0x4(%rcx,%rdx,1),%ymm7
  360e34:	c4 e2 45 08 d2       	vpsignb %ymm2,%ymm7,%ymm2
  360e39:	c4 e2 4d 04 d2       	vpmaddubsw %ymm2,%ymm6,%ymm2
  360e3e:	c4 e2 65 08 f3       	vpsignb %ymm3,%ymm3,%ymm6
  360e43:	c4 e2 7d 59 7c 11 0c 	vpbroadcastq 0xc(%rcx,%rdx,1),%ymm7
  360e4a:	c4 e2 45 08 db       	vpsignb %ymm3,%ymm7,%ymm3
  360e4f:	c4 e2 4d 04 db       	vpmaddubsw %ymm3,%ymm6,%ymm3
  360e54:	c5 e5 f5 d9          	vpmaddwd %ymm1,%ymm3,%ymm3
  360e58:	c5 ed f5 d1          	vpmaddwd %ymm1,%ymm2,%ymm2
  360e5c:	c5 e5 fe d2          	vpaddd %ymm2,%ymm3,%ymm2
  360e60:	c4 e2 5d 08 dc       	vpsignb %ymm4,%ymm4,%ymm3
  360e65:	c4 e2 7d 59 74 11 14 	vpbroadcastq 0x14(%rcx,%rdx,1),%ymm6
  360e6c:	c4 e2 4d 08 e4       	vpsignb %ymm4,%ymm6,%ymm4
  360e71:	c4 e2 65 04 dc       	vpmaddubsw %ymm4,%ymm3,%ymm3
  360e76:	c5 e5 f5 d9          	vpmaddwd %ymm1,%ymm3,%ymm3
  360e7a:	c5 ed fe d3          	vpaddd %ymm3,%ymm2,%ymm2
  360e7e:	c4 e2 55 08 dd       	vpsignb %ymm5,%ymm5,%ymm3
  360e83:	c4 e2 7d 59 64 11 1c 	vpbroadcastq 0x1c(%rcx,%rdx,1),%ymm4
  360e8a:	c4 e2 5d 08 e5       	vpsignb %ymm5,%ymm4,%ymm4
  360e8f:	c4 e2 65 04 dc       	vpmaddubsw %ymm4,%ymm3,%ymm3
  360e94:	c5 e5 f5 d9          	vpmaddwd %ymm1,%ymm3,%ymm3
  360e98:	c5 ed fe d3          	vpaddd %ymm3,%ymm2,%ymm2
  360e9c:	c4 e3 7d 39 d3 01    	vextracti128 $0x1,%ymm2,%xmm3
  360ea2:	c4 e2 69 02 d3       	vphaddd %xmm3,%xmm2,%xmm2
  360ea7:	c5 f8 5b d2          	vcvtdq2ps %xmm2,%xmm2
  360eab:	c5 e8 59 14 96       	vmulps (%rsi,%rdx,4),%xmm2,%xmm2
  360eb0:	62 f1 6c 18 59 14 11 	vmulps (%rcx,%rdx,1){1to4},%xmm2,%xmm2
  360eb7:	c5 f8 58 c2          	vaddps %xmm2,%xmm0,%xmm0
  360ebb:	48 83 c2 24          	add    $0x24,%rdx
  360ebf:	48 39 d0             	cmp    %rdx,%rax
  360ec2:	0f 85 48 ff ff ff    	jne    360e10 <camelid::inference::q8_0_packed_rows4_dot_i8_matmul+0x4c0>
  360ec8:	c5 f8 11 07          	vmovups %xmm0,(%rdi)
  360ecc:	48 83 c4 08          	add    $0x8,%rsp
  360ed0:	5b                   	pop    %rbx
  360ed1:	41 5c                	pop    %r12
  360ed3:	41 5d                	pop    %r13
  360ed5:	41 5e                	pop    %r14
  360ed7:	41 5f                	pop    %r15
  360ed9:	5d                   	pop    %rbp
  360eda:	c5 f8 77             	vzeroupper
  360edd:	c3                   	ret
  360ede:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  360ee4:	48 83 c4 08          	add    $0x8,%rsp
  360ee8:	5b                   	pop    %rbx
  360ee9:	41 5c                	pop    %r12
  360eeb:	41 5d                	pop    %r13
  360eed:	41 5e                	pop    %r14
  360eef:	41 5f                	pop    %r15
  360ef1:	5d                   	pop    %rbp
  360ef2:	e9 59 ac fe ff       	jmp    34bb50 <camelid::inference::q8_0_packed_rows4_dot>
  360ef7:	cc                   	int3
  360ef8:	cc                   	int3
  360ef9:	cc                   	int3
  360efa:	cc                   	int3
  360efb:	cc                   	int3
  360efc:	cc                   	int3
  360efd:	cc                   	int3
  360efe:	cc                   	int3
  360eff:	cc                   	int3
