
/home/ubuntu/work/llama.cpp-clean-20260517/build/bin/libggml-cpu.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

000000000005d610 <(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply(int, void const*, void const*, float*, int)>:
   5d610:	49 89 ca             	mov    %rcx,%r10
   5d613:	85 ff                	test   %edi,%edi
   5d615:	0f 8e fe 02 00 00    	jle    5d919 <(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply(int, void const*, void const*, float*, int)+0x309>
   5d61b:	48 89 f1             	mov    %rsi,%rcx
   5d61e:	69 f7 20 07 00 00    	imul   $0x720,%edi,%esi
   5d624:	48 89 d0             	mov    %rdx,%rax
   5d627:	4c 63 c7             	movslq %edi,%r8
   5d62a:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
   5d62e:	41 b9 80 ff ff ff    	mov    $0xffffff80,%r9d
   5d634:	4c 8b 1d 95 d8 0f 00 	mov    0xfd895(%rip),%r11        # 15aed0 <ggml_table_f32_f16@@Base-0x1450>
   5d63b:	c5 e1 ef db          	vpxor  %xmm3,%xmm3,%xmm3
   5d63f:	62 f1 7c 48 28 ec    	vmovaps %zmm4,%zmm5
   5d645:	62 f1 7c 48 28 f4    	vmovaps %zmm4,%zmm6
   5d64b:	62 f1 7c 48 28 fc    	vmovaps %zmm4,%zmm7
   5d651:	48 63 f6             	movslq %esi,%rsi
   5d654:	62 d2 7d 48 7a c9    	vpbroadcastb %r9d,%zmm1
   5d65a:	48 01 d6             	add    %rdx,%rsi
   5d65d:	69 d7 c0 04 00 00    	imul   $0x4c0,%edi,%edx
   5d663:	49 69 f8 60 02 00 00 	imul   $0x260,%r8,%rdi
   5d66a:	4d 6b c0 22          	imul   $0x22,%r8,%r8
   5d66e:	48 63 d2             	movslq %edx,%rdx
   5d671:	48 01 c2             	add    %rax,%rdx
   5d674:	48 01 c7             	add    %rax,%rdi
   5d677:	49 01 c8             	add    %rcx,%r8
   5d67a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   5d680:	62 72 7d 48 58 b9 02 	vpbroadcastd 0x2(%rcx),%zmm15
   5d687:	00 00 00 
   5d68a:	62 f1 7d 48 6f c3    	vmovdqa32 %zmm3,%zmm0
   5d690:	62 72 7d 48 58 b1 06 	vpbroadcastd 0x6(%rcx),%zmm14
   5d697:	00 00 00 
   5d69a:	62 e2 7d 48 13 40 10 	vcvtph2ps 0x200(%rax),%zmm16
   5d6a1:	48 83 c1 22          	add    $0x22,%rcx
   5d6a5:	62 72 7d 48 58 69 fa 	vpbroadcastd -0x18(%rcx),%zmm13
   5d6ac:	62 72 7d 48 58 61 fb 	vpbroadcastd -0x14(%rcx),%zmm12
   5d6b3:	48 05 60 02 00 00    	add    $0x260,%rax
   5d6b9:	48 81 c6 60 02 00 00 	add    $0x260,%rsi
   5d6c0:	62 71 05 48 fc f9    	vpaddb %zmm1,%zmm15,%zmm15
   5d6c6:	62 71 0d 48 fc f1    	vpaddb %zmm1,%zmm14,%zmm14
   5d6cc:	62 72 7d 48 58 59 fc 	vpbroadcastd -0x10(%rcx),%zmm11
   5d6d3:	62 72 7d 48 58 51 fd 	vpbroadcastd -0xc(%rcx),%zmm10
   5d6da:	62 f2 05 48 50 80 a0 	vpdpbusd -0x260(%rax),%zmm15,%zmm0
   5d6e1:	fd ff ff 
   5d6e4:	62 71 15 48 fc e9    	vpaddb %zmm1,%zmm13,%zmm13
   5d6ea:	62 71 1d 48 fc e1    	vpaddb %zmm1,%zmm12,%zmm12
   5d6f0:	62 72 7d 48 58 49 fe 	vpbroadcastd -0x8(%rcx),%zmm9
   5d6f7:	62 71 25 48 fc d9    	vpaddb %zmm1,%zmm11,%zmm11
   5d6fd:	62 71 2d 48 fc d1    	vpaddb %zmm1,%zmm10,%zmm10
   5d703:	62 72 7d 48 58 41 ff 	vpbroadcastd -0x4(%rcx),%zmm8
   5d70a:	44 0f b7 49 de       	movzwl -0x22(%rcx),%r9d
   5d70f:	62 71 35 48 fc c9    	vpaddb %zmm1,%zmm9,%zmm9
   5d715:	48 81 c2 60 02 00 00 	add    $0x260,%rdx
   5d71c:	48 81 c7 60 02 00 00 	add    $0x260,%rdi
   5d723:	62 71 3d 48 fc c1    	vpaddb %zmm1,%zmm8,%zmm8
   5d729:	62 92 7d 48 18 14 8b 	vbroadcastss (%r11,%r9,4),%zmm2
   5d730:	62 a1 6c 48 59 c0    	vmulps %zmm16,%zmm2,%zmm16
   5d736:	62 f2 0d 48 50 80 e0 	vpdpbusd -0x220(%rax),%zmm14,%zmm0
   5d73d:	fd ff ff 
   5d740:	62 f2 15 48 50 80 20 	vpdpbusd -0x1e0(%rax),%zmm13,%zmm0
   5d747:	fe ff ff 
   5d74a:	62 f2 1d 48 50 80 60 	vpdpbusd -0x1a0(%rax),%zmm12,%zmm0
   5d751:	fe ff ff 
   5d754:	62 f2 25 48 50 80 a0 	vpdpbusd -0x160(%rax),%zmm11,%zmm0
   5d75b:	fe ff ff 
   5d75e:	62 f2 2d 48 50 80 e0 	vpdpbusd -0x120(%rax),%zmm10,%zmm0
   5d765:	fe ff ff 
   5d768:	62 f2 35 48 50 80 20 	vpdpbusd -0xe0(%rax),%zmm9,%zmm0
   5d76f:	ff ff ff 
   5d772:	62 f2 3d 48 50 80 60 	vpdpbusd -0xa0(%rax),%zmm8,%zmm0
   5d779:	ff ff ff 
   5d77c:	62 f1 7d 48 fa 40 ff 	vpsubd -0x40(%rax),%zmm0,%zmm0
   5d783:	62 f1 7c 48 5b c0    	vcvtdq2ps %zmm0,%zmm0
   5d789:	62 b2 7d 48 b8 f8    	vfmadd231ps %zmm16,%zmm0,%zmm7
   5d78f:	62 f1 7d 48 6f c3    	vmovdqa32 %zmm3,%zmm0
   5d795:	62 e2 7d 48 13 47 fd 	vcvtph2ps -0x60(%rdi),%zmm16
   5d79c:	62 f2 05 48 50 87 a0 	vpdpbusd -0x260(%rdi),%zmm15,%zmm0
   5d7a3:	fd ff ff 
   5d7a6:	62 a1 6c 48 59 c0    	vmulps %zmm16,%zmm2,%zmm16
   5d7ac:	62 f2 0d 48 50 87 e0 	vpdpbusd -0x220(%rdi),%zmm14,%zmm0
   5d7b3:	fd ff ff 
   5d7b6:	62 f2 15 48 50 87 20 	vpdpbusd -0x1e0(%rdi),%zmm13,%zmm0
   5d7bd:	fe ff ff 
   5d7c0:	62 f2 1d 48 50 87 60 	vpdpbusd -0x1a0(%rdi),%zmm12,%zmm0
   5d7c7:	fe ff ff 
   5d7ca:	62 f2 25 48 50 87 a0 	vpdpbusd -0x160(%rdi),%zmm11,%zmm0
   5d7d1:	fe ff ff 
   5d7d4:	62 f2 2d 48 50 87 e0 	vpdpbusd -0x120(%rdi),%zmm10,%zmm0
   5d7db:	fe ff ff 
   5d7de:	62 f2 35 48 50 87 20 	vpdpbusd -0xe0(%rdi),%zmm9,%zmm0
   5d7e5:	ff ff ff 
   5d7e8:	62 f2 3d 48 50 87 60 	vpdpbusd -0xa0(%rdi),%zmm8,%zmm0
   5d7ef:	ff ff ff 
   5d7f2:	62 f1 7d 48 fa 47 ff 	vpsubd -0x40(%rdi),%zmm0,%zmm0
   5d7f9:	62 f1 7c 48 5b c0    	vcvtdq2ps %zmm0,%zmm0
   5d7ff:	62 b2 7d 48 b8 f0    	vfmadd231ps %zmm16,%zmm0,%zmm6
   5d805:	62 f1 7d 48 6f c3    	vmovdqa32 %zmm3,%zmm0
   5d80b:	62 e2 7d 48 13 42 fd 	vcvtph2ps -0x60(%rdx),%zmm16
   5d812:	62 f2 05 48 50 82 a0 	vpdpbusd -0x260(%rdx),%zmm15,%zmm0
   5d819:	fd ff ff 
   5d81c:	62 a1 6c 48 59 c0    	vmulps %zmm16,%zmm2,%zmm16
   5d822:	62 f2 0d 48 50 82 e0 	vpdpbusd -0x220(%rdx),%zmm14,%zmm0
   5d829:	fd ff ff 
   5d82c:	62 f2 15 48 50 82 20 	vpdpbusd -0x1e0(%rdx),%zmm13,%zmm0
   5d833:	fe ff ff 
   5d836:	62 f2 1d 48 50 82 60 	vpdpbusd -0x1a0(%rdx),%zmm12,%zmm0
   5d83d:	fe ff ff 
   5d840:	62 f2 25 48 50 82 a0 	vpdpbusd -0x160(%rdx),%zmm11,%zmm0
   5d847:	fe ff ff 
   5d84a:	62 f2 2d 48 50 82 e0 	vpdpbusd -0x120(%rdx),%zmm10,%zmm0
   5d851:	fe ff ff 
   5d854:	62 f2 35 48 50 82 20 	vpdpbusd -0xe0(%rdx),%zmm9,%zmm0
   5d85b:	ff ff ff 
   5d85e:	62 f2 3d 48 50 82 60 	vpdpbusd -0xa0(%rdx),%zmm8,%zmm0
   5d865:	ff ff ff 
   5d868:	62 f1 7d 48 fa 42 ff 	vpsubd -0x40(%rdx),%zmm0,%zmm0
   5d86f:	62 f1 7c 48 5b c0    	vcvtdq2ps %zmm0,%zmm0
   5d875:	62 b2 7d 48 b8 e8    	vfmadd231ps %zmm16,%zmm0,%zmm5
   5d87b:	62 f1 7d 48 6f c3    	vmovdqa32 %zmm3,%zmm0
   5d881:	62 e2 7d 48 13 46 fd 	vcvtph2ps -0x60(%rsi),%zmm16
   5d888:	62 f2 05 48 50 86 a0 	vpdpbusd -0x260(%rsi),%zmm15,%zmm0
   5d88f:	fd ff ff 
   5d892:	62 b1 6c 48 59 d0    	vmulps %zmm16,%zmm2,%zmm2
   5d898:	62 f2 0d 48 50 86 e0 	vpdpbusd -0x220(%rsi),%zmm14,%zmm0
   5d89f:	fd ff ff 
   5d8a2:	62 f2 15 48 50 86 20 	vpdpbusd -0x1e0(%rsi),%zmm13,%zmm0
   5d8a9:	fe ff ff 
   5d8ac:	62 f2 1d 48 50 86 60 	vpdpbusd -0x1a0(%rsi),%zmm12,%zmm0
   5d8b3:	fe ff ff 
   5d8b6:	62 f2 25 48 50 86 a0 	vpdpbusd -0x160(%rsi),%zmm11,%zmm0
   5d8bd:	fe ff ff 
   5d8c0:	62 f2 2d 48 50 86 e0 	vpdpbusd -0x120(%rsi),%zmm10,%zmm0
   5d8c7:	fe ff ff 
   5d8ca:	62 f2 35 48 50 86 20 	vpdpbusd -0xe0(%rsi),%zmm9,%zmm0
   5d8d1:	ff ff ff 
   5d8d4:	62 f2 3d 48 50 86 60 	vpdpbusd -0xa0(%rsi),%zmm8,%zmm0
   5d8db:	ff ff ff 
   5d8de:	62 f1 7d 48 fa 46 ff 	vpsubd -0x40(%rsi),%zmm0,%zmm0
   5d8e5:	62 f1 7c 48 5b c0    	vcvtdq2ps %zmm0,%zmm0
   5d8eb:	62 f2 7d 48 b8 e2    	vfmadd231ps %zmm2,%zmm0,%zmm4
   5d8f1:	49 39 c8             	cmp    %rcx,%r8
   5d8f4:	0f 85 86 fd ff ff    	jne    5d680 <(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply(int, void const*, void const*, float*, int)+0x70>
   5d8fa:	62 d1 7c 48 11 3a    	vmovups %zmm7,(%r10)
   5d900:	62 d1 7c 48 11 72 01 	vmovups %zmm6,0x40(%r10)
   5d907:	62 d1 7c 48 11 6a 02 	vmovups %zmm5,0x80(%r10)
   5d90e:	62 d1 7c 48 11 62 03 	vmovups %zmm4,0xc0(%r10)
   5d915:	c5 f8 77             	vzeroupper
   5d918:	c3                   	ret
   5d919:	c5 d8 57 e4          	vxorps %xmm4,%xmm4,%xmm4
   5d91d:	62 f1 7c 48 28 ec    	vmovaps %zmm4,%zmm5
   5d923:	62 f1 7c 48 28 f4    	vmovaps %zmm4,%zmm6
   5d929:	62 f1 7c 48 28 fc    	vmovaps %zmm4,%zmm7
   5d92f:	eb c9                	jmp    5d8fa <(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply(int, void const*, void const*, float*, int)+0x2ea>
   5d931:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
   5d938:	00 00 00 00 
   5d93c:	0f 1f 40 00          	nopl   0x0(%rax)
