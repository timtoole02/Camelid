
/home/ubuntu/work/llama.cpp-clean-20260517/build/bin/libggml-cpu.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

000000000006c8d0 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]>:
   6c8d0:	55                   	push   %rbp
   6c8d1:	48 89 e5             	mov    %rsp,%rbp
   6c8d4:	41 57                	push   %r15
   6c8d6:	41 56                	push   %r14
   6c8d8:	41 55                	push   %r13
   6c8da:	41 54                	push   %r12
   6c8dc:	53                   	push   %rbx
   6c8dd:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
   6c8e1:	48 81 ec 80 05 00 00 	sub    $0x580,%rsp
   6c8e8:	89 bc 24 00 05 00 00 	mov    %edi,0x500(%rsp)
   6c8ef:	89 b4 24 6c 05 00 00 	mov    %esi,0x56c(%rsp)
   6c8f6:	48 89 8c 24 70 05 00 	mov    %rcx,0x570(%rsp)
   6c8fd:	00 
   6c8fe:	4c 89 84 24 78 05 00 	mov    %r8,0x578(%rsp)
   6c905:	00 
   6c906:	83 ff 20             	cmp    $0x20,%edi
   6c909:	0f 8f c3 41 00 00    	jg     70ad2 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x4202>
   6c90f:	48 63 de             	movslq %esi,%rbx
   6c912:	c7 84 24 c0 04 00 00 	movl   $0x0,0x4c0(%rsp)
   6c919:	00 00 00 00 
   6c91d:	49 89 d7             	mov    %rdx,%r15
   6c920:	6b c3 22             	imul   $0x22,%ebx,%eax
   6c923:	83 bc 24 00 05 00 00 	cmpl   $0x10,0x500(%rsp)
   6c92a:	10 
   6c92b:	48 89 9c 24 48 05 00 	mov    %rbx,0x548(%rsp)
   6c932:	00 
   6c933:	89 84 24 58 05 00 00 	mov    %eax,0x558(%rsp)
   6c93a:	0f 8f 78 13 00 00    	jg     6dcb8 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x13e8>
   6c940:	8b 8c 24 6c 05 00 00 	mov    0x56c(%rsp),%ecx
   6c947:	85 c9                	test   %ecx,%ecx
   6c949:	0f 8e c4 06 00 00    	jle    6d013 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x743>
   6c94f:	89 c8                	mov    %ecx,%eax
   6c951:	48 8b 9c 24 48 05 00 	mov    0x548(%rsp),%rbx
   6c958:	00 
   6c959:	48 8b bc 24 70 05 00 	mov    0x570(%rsp),%rdi
   6c960:	00 
   6c961:	4d 89 fd             	mov    %r15,%r13
   6c964:	c1 e0 04             	shl    $0x4,%eax
   6c967:	89 84 24 30 04 00 00 	mov    %eax,0x430(%rsp)
   6c96e:	44 89 c8             	mov    %r9d,%eax
   6c971:	4d 63 c9             	movslq %r9d,%r9
   6c974:	c1 e0 04             	shl    $0x4,%eax
   6c977:	48 98                	cltq
   6c979:	48 89 84 24 00 04 00 	mov    %rax,0x400(%rsp)
   6c980:	00 
   6c981:	48 83 c0 10          	add    $0x10,%rax
   6c985:	48 89 84 24 d8 03 00 	mov    %rax,0x3d8(%rsp)
   6c98c:	00 
   6c98d:	48 69 c3 60 02 00 00 	imul   $0x260,%rbx,%rax
   6c994:	48 6b db 22          	imul   $0x22,%rbx,%rbx
   6c998:	48 01 f8             	add    %rdi,%rax
   6c99b:	48 89 84 24 80 04 00 	mov    %rax,0x480(%rsp)
   6c9a2:	00 
   6c9a3:	49 8d 04 1f          	lea    (%r15,%rbx,1),%rax
   6c9a7:	48 89 84 24 40 04 00 	mov    %rax,0x440(%rsp)
   6c9ae:	00 
   6c9af:	8d 04 09             	lea    (%rcx,%rcx,1),%eax
   6c9b2:	48 63 f8             	movslq %eax,%rdi
   6c9b5:	89 84 24 28 04 00 00 	mov    %eax,0x428(%rsp)
   6c9bc:	01 c8                	add    %ecx,%eax
   6c9be:	48 98                	cltq
   6c9c0:	48 89 bc 24 f8 03 00 	mov    %rdi,0x3f8(%rsp)
   6c9c7:	00 
   6c9c8:	4c 6b e7 22          	imul   $0x22,%rdi,%r12
   6c9cc:	48 89 84 24 e0 03 00 	mov    %rax,0x3e0(%rsp)
   6c9d3:	00 
   6c9d4:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6c9d8:	4d 01 fc             	add    %r15,%r12
   6c9db:	4c 01 f8             	add    %r15,%rax
   6c9de:	48 89 84 24 50 05 00 	mov    %rax,0x550(%rsp)
   6c9e5:	00 
   6c9e6:	8d 04 8d 00 00 00 00 	lea    0x0(,%rcx,4),%eax
   6c9ed:	48 63 f8             	movslq %eax,%rdi
   6c9f0:	89 84 24 e8 03 00 00 	mov    %eax,0x3e8(%rsp)
   6c9f7:	01 c8                	add    %ecx,%eax
   6c9f9:	48 98                	cltq
   6c9fb:	4c 6b f7 22          	imul   $0x22,%rdi,%r14
   6c9ff:	48 89 bc 24 d0 03 00 	mov    %rdi,0x3d0(%rsp)
   6ca06:	00 
   6ca07:	48 89 84 24 c8 03 00 	mov    %rax,0x3c8(%rsp)
   6ca0e:	00 
   6ca0f:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6ca13:	4b 8d 3c 37          	lea    (%r15,%r14,1),%rdi
   6ca17:	4e 8d 34 8d 00 00 00 	lea    0x0(,%r9,4),%r14
   6ca1e:	00 
   6ca1f:	4c 01 f8             	add    %r15,%rax
   6ca22:	48 89 bc 24 20 04 00 	mov    %rdi,0x420(%rsp)
   6ca29:	00 
   6ca2a:	48 89 84 24 f0 03 00 	mov    %rax,0x3f0(%rsp)
   6ca31:	00 
   6ca32:	48 8d 3d 4f e4 0e 00 	lea    0xee44f(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6ca39:	e8 02 85 fa ff       	call   14f40 <__tls_get_addr@plt>
   6ca3e:	48 8d 88 00 a8 00 00 	lea    0xa800(%rax),%rcx
   6ca45:	48 63 84 24 00 05 00 	movslq 0x500(%rsp),%rax
   6ca4c:	00 
   6ca4d:	48 89 8c 24 60 05 00 	mov    %rcx,0x560(%rsp)
   6ca54:	00 
   6ca55:	48 c1 e0 06          	shl    $0x6,%rax
   6ca59:	48 8d 3c 01          	lea    (%rcx,%rax,1),%rdi
   6ca5d:	48 8d 84 01 00 08 00 	lea    0x800(%rcx,%rax,1),%rax
   6ca64:	00 
   6ca65:	48 89 bc 24 38 04 00 	mov    %rdi,0x438(%rsp)
   6ca6c:	00 
   6ca6d:	48 89 84 24 40 05 00 	mov    %rax,0x540(%rsp)
   6ca74:	00 
   6ca75:	c4 e2 7b 49 e0       	tilezero %tmm4
   6ca7a:	c4 e2 7b 49 f0       	tilezero %tmm6
   6ca7f:	44 8b 84 24 c0 04 00 	mov    0x4c0(%rsp),%r8d
   6ca86:	00 
   6ca87:	45 85 c0             	test   %r8d,%r8d
   6ca8a:	0f 85 98 05 00 00    	jne    6d028 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x758>
   6ca90:	48 8b 8c 24 70 05 00 	mov    0x570(%rsp),%rcx
   6ca97:	00 
   6ca98:	b8 40 00 00 00       	mov    $0x40,%eax
   6ca9d:	c4 e2 7b 4b 04 01    	tileloadd (%rcx,%rax,1),%tmm0
   6caa3:	48 8b b4 24 80 04 00 	mov    0x480(%rsp),%rsi
   6caaa:	00 
   6caab:	c4 e2 7b 4b 0c 06    	tileloadd (%rsi,%rax,1),%tmm1
   6cab1:	83 bc 24 00 05 00 00 	cmpl   $0x10,0x500(%rsp)
   6cab8:	10 
   6cab9:	0f 85 81 05 00 00    	jne    6d040 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x770>
   6cabf:	49 8d 47 02          	lea    0x2(%r15),%rax
   6cac3:	48 63 bc 24 58 05 00 	movslq 0x558(%rsp),%rdi
   6caca:	00 
   6cacb:	c4 e2 7b 4b 14 38    	tileloadd (%rax,%rdi,1),%tmm2
   6cad1:	c4 e2 7b 5e e2       	tdpbssd %tmm0,%tmm2,%tmm4
   6cad6:	c4 e2 73 5e f2       	tdpbssd %tmm1,%tmm2,%tmm6
   6cadb:	48 8d 3d a6 e3 0e 00 	lea    0xee3a6(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6cae2:	e8 59 84 fa ff       	call   14f40 <__tls_get_addr@plt>
   6cae7:	bf 40 00 00 00       	mov    $0x40,%edi
   6caec:	48 05 00 a8 00 00    	add    $0xa800,%rax
   6caf2:	c4 e2 7a 4b 24 38    	tilestored %tmm4,(%rax,%rdi,1)
   6caf8:	48 05 00 08 00 00    	add    $0x800,%rax
   6cafe:	c4 e2 7a 4b 34 38    	tilestored %tmm6,(%rax,%rdi,1)
   6cb04:	8b b4 24 00 05 00 00 	mov    0x500(%rsp),%esi
   6cb0b:	85 f6                	test   %esi,%esi
   6cb0d:	0f 8e ed 00 00 00    	jle    6cc00 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x330>
   6cb13:	48 8b 84 24 70 05 00 	mov    0x570(%rsp),%rax
   6cb1a:	00 
   6cb1b:	4c 8b 84 24 78 05 00 	mov    0x578(%rsp),%r8
   6cb22:	00 
   6cb23:	4c 89 ff             	mov    %r15,%rdi
   6cb26:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
   6cb2a:	4c 8b 25 9f e3 0e 00 	mov    0xee39f(%rip),%r12        # 15aed0 <ggml_table_f32_f16@@Base-0x1450>
   6cb31:	48 8b 8c 24 38 04 00 	mov    0x438(%rsp),%rcx
   6cb38:	00 
   6cb39:	62 f2 7d 48 13 58 10 	vcvtph2ps 0x200(%rax),%zmm3
   6cb40:	48 8b 84 24 60 05 00 	mov    0x560(%rsp),%rax
   6cb47:	00 
   6cb48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
   6cb4f:	00 
   6cb50:	0f b7 17             	movzwl (%rdi),%edx
   6cb53:	62 f1 7c 48 5b 00    	vcvtdq2ps (%rax),%zmm0
   6cb59:	48 83 c0 40          	add    $0x40,%rax
   6cb5d:	48 01 df             	add    %rbx,%rdi
   6cb60:	62 d1 64 58 59 0c 94 	vmulps (%r12,%rdx,4){1to16},%zmm3,%zmm1
   6cb67:	62 f2 6d 48 98 c1    	vfmadd132ps %zmm1,%zmm2,%zmm0
   6cb6d:	62 d1 7c 48 11 00    	vmovups %zmm0,(%r8)
   6cb73:	4d 01 f0             	add    %r14,%r8
   6cb76:	48 39 c8             	cmp    %rcx,%rax
   6cb79:	75 d5                	jne    6cb50 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x280>
   6cb7b:	48 8b 84 24 80 04 00 	mov    0x480(%rsp),%rax
   6cb82:	00 
   6cb83:	62 f2 7d 48 13 58 10 	vcvtph2ps 0x200(%rax),%zmm3
   6cb8a:	62 f1 7c 48 29 5c 24 	vmovaps %zmm3,0x440(%rsp)
   6cb91:	11 
   6cb92:	c5 f8 77             	vzeroupper
   6cb95:	48 8d 3d ec e2 0e 00 	lea    0xee2ec(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6cb9c:	e8 9f 83 fa ff       	call   14f40 <__tls_get_addr@plt>
   6cba1:	48 8b bc 24 78 05 00 	mov    0x578(%rsp),%rdi
   6cba8:	00 
   6cba9:	48 8b 8c 24 40 05 00 	mov    0x540(%rsp),%rcx
   6cbb0:	00 
   6cbb1:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
   6cbb5:	62 f1 7c 48 28 5c 24 	vmovaps 0x440(%rsp),%zmm3
   6cbbc:	11 
   6cbbd:	48 83 c7 40          	add    $0x40,%rdi
   6cbc1:	48 05 00 b0 00 00    	add    $0xb000,%rax
   6cbc7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
   6cbce:	00 00 
   6cbd0:	41 0f b7 55 00       	movzwl 0x0(%r13),%edx
   6cbd5:	62 f1 7c 48 5b 00    	vcvtdq2ps (%rax),%zmm0
   6cbdb:	48 83 c0 40          	add    $0x40,%rax
   6cbdf:	49 01 dd             	add    %rbx,%r13
   6cbe2:	62 d1 64 58 59 0c 94 	vmulps (%r12,%rdx,4){1to16},%zmm3,%zmm1
   6cbe9:	62 f2 6d 48 98 c1    	vfmadd132ps %zmm1,%zmm2,%zmm0
   6cbef:	62 f1 7c 48 11 07    	vmovups %zmm0,(%rdi)
   6cbf5:	4c 01 f7             	add    %r14,%rdi
   6cbf8:	48 39 c1             	cmp    %rax,%rcx
   6cbfb:	75 d3                	jne    6cbd0 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x300>
   6cbfd:	c5 f8 77             	vzeroupper
   6cc00:	8b 8c 24 c0 04 00 00 	mov    0x4c0(%rsp),%ecx
   6cc07:	85 c9                	test   %ecx,%ecx
   6cc09:	0f 85 49 09 00 00    	jne    6d558 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc88>
   6cc0f:	8b b4 24 6c 05 00 00 	mov    0x56c(%rsp),%esi
   6cc16:	83 fe 01             	cmp    $0x1,%esi
   6cc19:	0f 84 f4 03 00 00    	je     6d013 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x743>
   6cc1f:	8d 46 01             	lea    0x1(%rsi),%eax
   6cc22:	48 8b 8c 24 70 05 00 	mov    0x570(%rsp),%rcx
   6cc29:	00 
   6cc2a:	48 c7 84 24 70 05 00 	movq   $0x1,0x570(%rsp)
   6cc31:	00 01 00 00 00 
   6cc36:	4d 8d 4f 22          	lea    0x22(%r15),%r9
   6cc3a:	48 98                	cltq
   6cc3c:	48 69 c0 60 02 00 00 	imul   $0x260,%rax,%rax
   6cc43:	4c 8d 99 60 02 00 00 	lea    0x260(%rcx),%r11
   6cc4a:	4c 89 9c 24 40 04 00 	mov    %r11,0x440(%rsp)
   6cc51:	00 
   6cc52:	48 8d bc 01 00 02 00 	lea    0x200(%rcx,%rax,1),%rdi
   6cc59:	00 
   6cc5a:	48 01 c8             	add    %rcx,%rax
   6cc5d:	48 89 84 24 80 04 00 	mov    %rax,0x480(%rsp)
   6cc64:	00 
   6cc65:	48 8b 84 24 48 05 00 	mov    0x548(%rsp),%rax
   6cc6c:	00 
   6cc6d:	48 89 bc 24 50 05 00 	mov    %rdi,0x550(%rsp)
   6cc74:	00 
   6cc75:	48 8b bc 24 78 05 00 	mov    0x578(%rsp),%rdi
   6cc7c:	00 
   6cc7d:	48 ff c0             	inc    %rax
   6cc80:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6cc84:	4c 01 f8             	add    %r15,%rax
   6cc87:	48 89 84 24 c0 03 00 	mov    %rax,0x3c0(%rsp)
   6cc8e:	00 
   6cc8f:	48 8b 84 24 f8 03 00 	mov    0x3f8(%rsp),%rax
   6cc96:	00 
   6cc97:	48 ff c0             	inc    %rax
   6cc9a:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6cc9e:	4c 01 f8             	add    %r15,%rax
   6cca1:	48 89 84 24 b8 03 00 	mov    %rax,0x3b8(%rsp)
   6cca8:	00 
   6cca9:	48 8b 84 24 e0 03 00 	mov    0x3e0(%rsp),%rax
   6ccb0:	00 
   6ccb1:	48 ff c0             	inc    %rax
   6ccb4:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6ccb8:	4c 01 f8             	add    %r15,%rax
   6ccbb:	48 89 84 24 a8 03 00 	mov    %rax,0x3a8(%rsp)
   6ccc2:	00 
   6ccc3:	48 8b 84 24 d0 03 00 	mov    0x3d0(%rsp),%rax
   6ccca:	00 
   6cccb:	48 ff c0             	inc    %rax
   6ccce:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6ccd2:	4c 01 f8             	add    %r15,%rax
   6ccd5:	48 89 84 24 98 03 00 	mov    %rax,0x398(%rsp)
   6ccdc:	00 
   6ccdd:	48 8b 84 24 c8 03 00 	mov    0x3c8(%rsp),%rax
   6cce4:	00 
   6cce5:	48 ff c0             	inc    %rax
   6cce8:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6ccec:	4c 01 f8             	add    %r15,%rax
   6ccef:	48 89 84 24 90 03 00 	mov    %rax,0x390(%rsp)
   6ccf6:	00 
   6ccf7:	8b 84 24 28 04 00 00 	mov    0x428(%rsp),%eax
   6ccfe:	01 f0                	add    %esi,%eax
   6cd00:	01 c0                	add    %eax,%eax
   6cd02:	48 63 d0             	movslq %eax,%rdx
   6cd05:	48 8d 42 01          	lea    0x1(%rdx),%rax
   6cd09:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6cd0d:	4c 01 f8             	add    %r15,%rax
   6cd10:	48 89 84 24 88 03 00 	mov    %rax,0x388(%rsp)
   6cd17:	00 
   6cd18:	8d 04 f5 00 00 00 00 	lea    0x0(,%rsi,8),%eax
   6cd1f:	89 c1                	mov    %eax,%ecx
   6cd21:	29 f1                	sub    %esi,%ecx
   6cd23:	48 63 c9             	movslq %ecx,%rcx
   6cd26:	48 ff c1             	inc    %rcx
   6cd29:	48 6b c9 22          	imul   $0x22,%rcx,%rcx
   6cd2d:	4c 01 f9             	add    %r15,%rcx
   6cd30:	48 89 8c 24 80 03 00 	mov    %rcx,0x380(%rsp)
   6cd37:	00 
   6cd38:	48 63 c8             	movslq %eax,%rcx
   6cd3b:	01 f0                	add    %esi,%eax
   6cd3d:	48 98                	cltq
   6cd3f:	48 ff c1             	inc    %rcx
   6cd42:	48 ff c0             	inc    %rax
   6cd45:	48 6b c9 22          	imul   $0x22,%rcx,%rcx
   6cd49:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6cd4d:	4c 01 f9             	add    %r15,%rcx
   6cd50:	4c 01 f8             	add    %r15,%rax
   6cd53:	48 89 8c 24 78 03 00 	mov    %rcx,0x378(%rsp)
   6cd5a:	00 
   6cd5b:	48 8b 8c 24 60 05 00 	mov    0x560(%rsp),%rcx
   6cd62:	00 
   6cd63:	48 89 84 24 68 03 00 	mov    %rax,0x368(%rsp)
   6cd6a:	00 
   6cd6b:	8b 84 24 e8 03 00 00 	mov    0x3e8(%rsp),%eax
   6cd72:	01 f0                	add    %esi,%eax
   6cd74:	45 31 ed             	xor    %r13d,%r13d
   6cd77:	01 c0                	add    %eax,%eax
   6cd79:	48 98                	cltq
   6cd7b:	48 ff c0             	inc    %rax
   6cd7e:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6cd82:	4c 01 f8             	add    %r15,%rax
   6cd85:	48 89 84 24 50 03 00 	mov    %rax,0x350(%rsp)
   6cd8c:	00 
   6cd8d:	48 8b 84 24 00 04 00 	mov    0x400(%rsp),%rax
   6cd94:	00 
   6cd95:	48 8d 04 87          	lea    (%rdi,%rax,4),%rax
   6cd99:	48 89 84 24 a0 03 00 	mov    %rax,0x3a0(%rsp)
   6cda0:	00 
   6cda1:	48 63 84 24 c0 04 00 	movslq 0x4c0(%rsp),%rax
   6cda8:	00 
   6cda9:	48 c1 e0 06          	shl    $0x6,%rax
   6cdad:	48 8d b4 01 00 04 00 	lea    0x400(%rcx,%rax,1),%rsi
   6cdb4:	00 
   6cdb5:	48 8d 84 01 00 0c 00 	lea    0xc00(%rcx,%rax,1),%rax
   6cdbc:	00 
   6cdbd:	48 89 b4 24 b0 03 00 	mov    %rsi,0x3b0(%rsp)
   6cdc4:	00 
   6cdc5:	48 8b b4 24 d8 03 00 	mov    0x3d8(%rsp),%rsi
   6cdcc:	00 
   6cdcd:	48 89 84 24 60 03 00 	mov    %rax,0x360(%rsp)
   6cdd4:	00 
   6cdd5:	48 63 84 24 58 05 00 	movslq 0x558(%rsp),%rax
   6cddc:	00 
   6cddd:	48 8d 3c b7          	lea    (%rdi,%rsi,4),%rdi
   6cde1:	48 89 bc 24 70 03 00 	mov    %rdi,0x370(%rsp)
   6cde8:	00 
   6cde9:	48 89 84 24 f0 03 00 	mov    %rax,0x3f0(%rsp)
   6cdf0:	00 
   6cdf1:	44 8b a4 24 00 05 00 	mov    0x500(%rsp),%r12d
   6cdf8:	00 
   6cdf9:	4c 89 9c 24 58 05 00 	mov    %r11,0x558(%rsp)
   6ce00:	00 
   6ce01:	48 89 94 24 58 03 00 	mov    %rdx,0x358(%rsp)
   6ce08:	00 
   6ce09:	4c 89 bc 24 20 04 00 	mov    %r15,0x420(%rsp)
   6ce10:	00 
   6ce11:	49 89 df             	mov    %rbx,%r15
   6ce14:	4c 89 f3             	mov    %r14,%rbx
   6ce17:	4d 89 ce             	mov    %r9,%r14
   6ce1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   6ce20:	c4 e2 7b 49 e0       	tilezero %tmm4
   6ce25:	c4 e2 7b 49 f0       	tilezero %tmm6
   6ce2a:	8b 94 24 c0 04 00 00 	mov    0x4c0(%rsp),%edx
   6ce31:	85 d2                	test   %edx,%edx
   6ce33:	74 0a                	je     6ce3f <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x56f>
   6ce35:	c4 e2 7b 49 e8       	tilezero %tmm5
   6ce3a:	c4 e2 7b 49 f8       	tilezero %tmm7
   6ce3f:	48 8b bc 24 40 04 00 	mov    0x440(%rsp),%rdi
   6ce46:	00 
   6ce47:	b8 40 00 00 00       	mov    $0x40,%eax
   6ce4c:	c4 e2 7b 4b 04 07    	tileloadd (%rdi,%rax,1),%tmm0
   6ce52:	48 8b bc 24 80 04 00 	mov    0x480(%rsp),%rdi
   6ce59:	00 
   6ce5a:	c4 e2 7b 4b 0c 07    	tileloadd (%rdi,%rax,1),%tmm1
   6ce60:	41 83 fc 10          	cmp    $0x10,%r12d
   6ce64:	0f 85 2e 04 00 00    	jne    6d298 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9c8>
   6ce6a:	49 8d 46 02          	lea    0x2(%r14),%rax
   6ce6e:	48 8b 8c 24 f0 03 00 	mov    0x3f0(%rsp),%rcx
   6ce75:	00 
   6ce76:	c4 e2 7b 4b 14 08    	tileloadd (%rax,%rcx,1),%tmm2
   6ce7c:	c4 e2 7b 5e e2       	tdpbssd %tmm0,%tmm2,%tmm4
   6ce81:	c4 e2 73 5e f2       	tdpbssd %tmm1,%tmm2,%tmm6
   6ce86:	48 8d 3d fb df 0e 00 	lea    0xedffb(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6ce8d:	e8 ae 80 fa ff       	call   14f40 <__tls_get_addr@plt>
   6ce92:	b9 40 00 00 00       	mov    $0x40,%ecx
   6ce97:	48 05 00 a8 00 00    	add    $0xa800,%rax
   6ce9d:	c4 e2 7a 4b 24 08    	tilestored %tmm4,(%rax,%rcx,1)
   6cea3:	48 05 00 08 00 00    	add    $0x800,%rax
   6cea9:	c4 e2 7a 4b 34 08    	tilestored %tmm6,(%rax,%rcx,1)
   6ceaf:	45 85 e4             	test   %r12d,%r12d
   6ceb2:	0f 8e f7 00 00 00    	jle    6cfaf <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x6df>
   6ceb8:	48 8b 84 24 58 05 00 	mov    0x558(%rsp),%rax
   6cebf:	00 
   6cec0:	48 8b 94 24 78 05 00 	mov    0x578(%rsp),%rdx
   6cec7:	00 
   6cec8:	4d 89 f0             	mov    %r14,%r8
   6cecb:	48 8b 0d fe df 0e 00 	mov    0xedffe(%rip),%rcx        # 15aed0 <ggml_table_f32_f16@@Base-0x1450>
   6ced2:	48 8b bc 24 38 04 00 	mov    0x438(%rsp),%rdi
   6ced9:	00 
   6ceda:	62 f2 7d 48 13 50 10 	vcvtph2ps 0x200(%rax),%zmm2
   6cee1:	48 8b 84 24 60 05 00 	mov    0x560(%rsp),%rax
   6cee8:	00 
   6cee9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   6cef0:	41 0f b7 30          	movzwl (%r8),%esi
   6cef4:	62 f1 7c 48 5b 00    	vcvtdq2ps (%rax),%zmm0
   6cefa:	48 83 c0 40          	add    $0x40,%rax
   6cefe:	4d 01 f8             	add    %r15,%r8
   6cf01:	62 f1 6c 58 59 0c b1 	vmulps (%rcx,%rsi,4){1to16},%zmm2,%zmm1
   6cf08:	62 f2 75 48 a8 02    	vfmadd213ps (%rdx),%zmm1,%zmm0
   6cf0e:	62 f1 7c 48 11 02    	vmovups %zmm0,(%rdx)
   6cf14:	48 01 da             	add    %rbx,%rdx
   6cf17:	48 39 c7             	cmp    %rax,%rdi
   6cf1a:	75 d4                	jne    6cef0 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x620>
   6cf1c:	48 89 8c 24 00 04 00 	mov    %rcx,0x400(%rsp)
   6cf23:	00 
   6cf24:	48 8b 84 24 50 05 00 	mov    0x550(%rsp),%rax
   6cf2b:	00 
   6cf2c:	62 f2 7d 48 13 10    	vcvtph2ps (%rax),%zmm2
   6cf32:	62 f1 7c 48 29 54 24 	vmovaps %zmm2,0x500(%rsp)
   6cf39:	14 
   6cf3a:	c5 f8 77             	vzeroupper
   6cf3d:	48 8d 3d 44 df 0e 00 	lea    0xedf44(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6cf44:	e8 f7 7f fa ff       	call   14f40 <__tls_get_addr@plt>
   6cf49:	48 8b bc 24 78 05 00 	mov    0x578(%rsp),%rdi
   6cf50:	00 
   6cf51:	48 8b 8c 24 00 04 00 	mov    0x400(%rsp),%rcx
   6cf58:	00 
   6cf59:	4d 89 f0             	mov    %r14,%r8
   6cf5c:	62 f1 7c 48 28 54 24 	vmovaps 0x500(%rsp),%zmm2
   6cf63:	14 
   6cf64:	48 8d 57 40          	lea    0x40(%rdi),%rdx
   6cf68:	48 8b bc 24 40 05 00 	mov    0x540(%rsp),%rdi
   6cf6f:	00 
   6cf70:	48 05 00 b0 00 00    	add    $0xb000,%rax
   6cf76:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
   6cf7d:	00 00 00 
   6cf80:	41 0f b7 30          	movzwl (%r8),%esi
   6cf84:	62 f1 7c 48 5b 00    	vcvtdq2ps (%rax),%zmm0
   6cf8a:	48 83 c0 40          	add    $0x40,%rax
   6cf8e:	4d 01 f8             	add    %r15,%r8
   6cf91:	62 f1 6c 58 59 0c b1 	vmulps (%rcx,%rsi,4){1to16},%zmm2,%zmm1
   6cf98:	62 f2 75 48 a8 02    	vfmadd213ps (%rdx),%zmm1,%zmm0
   6cf9e:	62 f1 7c 48 11 02    	vmovups %zmm0,(%rdx)
   6cfa4:	48 01 da             	add    %rbx,%rdx
   6cfa7:	48 39 c7             	cmp    %rax,%rdi
   6cfaa:	75 d4                	jne    6cf80 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x6b0>
   6cfac:	c5 f8 77             	vzeroupper
   6cfaf:	8b 84 24 c0 04 00 00 	mov    0x4c0(%rsp),%eax
   6cfb6:	85 c0                	test   %eax,%eax
   6cfb8:	0f 85 1a 09 00 00    	jne    6d8d8 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x1008>
   6cfbe:	48 ff 84 24 70 05 00 	incq   0x570(%rsp)
   6cfc5:	00 
   6cfc6:	49 83 c6 22          	add    $0x22,%r14
   6cfca:	49 83 c5 22          	add    $0x22,%r13
   6cfce:	48 81 84 24 58 05 00 	addq   $0x260,0x558(%rsp)
   6cfd5:	00 60 02 00 00 
   6cfda:	48 81 84 24 50 05 00 	addq   $0x260,0x550(%rsp)
   6cfe1:	00 60 02 00 00 
   6cfe6:	48 81 84 24 40 04 00 	addq   $0x260,0x440(%rsp)
   6cfed:	00 60 02 00 00 
   6cff2:	48 8b 84 24 70 05 00 	mov    0x570(%rsp),%rax
   6cff9:	00 
   6cffa:	48 81 84 24 80 04 00 	addq   $0x260,0x480(%rsp)
   6d001:	00 60 02 00 00 
   6d006:	39 84 24 6c 05 00 00 	cmp    %eax,0x56c(%rsp)
   6d00d:	0f 8f 0d fe ff ff    	jg     6ce20 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x550>
   6d013:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
   6d017:	5b                   	pop    %rbx
   6d018:	41 5c                	pop    %r12
   6d01a:	41 5d                	pop    %r13
   6d01c:	41 5e                	pop    %r14
   6d01e:	41 5f                	pop    %r15
   6d020:	5d                   	pop    %rbp
   6d021:	c3                   	ret
   6d022:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   6d028:	c4 e2 7b 49 e8       	tilezero %tmm5
   6d02d:	c4 e2 7b 49 f8       	tilezero %tmm7
   6d032:	e9 59 fa ff ff       	jmp    6ca90 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x1c0>
   6d037:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
   6d03e:	00 00 
   6d040:	8b bc 24 00 05 00 00 	mov    0x500(%rsp),%edi
   6d047:	85 ff                	test   %edi,%edi
   6d049:	0f 8e 24 02 00 00    	jle    6d273 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a3>
   6d04f:	48 8d 3d 32 de 0e 00 	lea    0xede32(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6d056:	e8 e5 7e fa ff       	call   14f40 <__tls_get_addr@plt>
   6d05b:	c4 c1 7e 6f 67 02    	vmovdqu 0x2(%r15),%ymm4
   6d061:	8b b4 24 00 05 00 00 	mov    0x500(%rsp),%esi
   6d068:	48 8d b8 00 b8 00 00 	lea    0xb800(%rax),%rdi
   6d06f:	c5 fd 7f a0 00 b8 00 	vmovdqa %ymm4,0xb800(%rax)
   6d076:	00 
   6d077:	83 fe 01             	cmp    $0x1,%esi
   6d07a:	0f 84 f0 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d080:	48 8b 84 24 40 04 00 	mov    0x440(%rsp),%rax
   6d087:	00 
   6d088:	c5 fe 6f 68 02       	vmovdqu 0x2(%rax),%ymm5
   6d08d:	c5 fd 7f 6f 20       	vmovdqa %ymm5,0x20(%rdi)
   6d092:	83 fe 02             	cmp    $0x2,%esi
   6d095:	0f 84 d5 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d09b:	c4 c1 7e 6f 74 24 02 	vmovdqu 0x2(%r12),%ymm6
   6d0a2:	c5 fd 7f 77 40       	vmovdqa %ymm6,0x40(%rdi)
   6d0a7:	83 fe 03             	cmp    $0x3,%esi
   6d0aa:	0f 84 c0 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d0b0:	48 8b 84 24 50 05 00 	mov    0x550(%rsp),%rax
   6d0b7:	00 
   6d0b8:	c5 fe 6f 78 02       	vmovdqu 0x2(%rax),%ymm7
   6d0bd:	c5 fd 7f 7f 60       	vmovdqa %ymm7,0x60(%rdi)
   6d0c2:	83 fe 04             	cmp    $0x4,%esi
   6d0c5:	0f 84 a5 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d0cb:	48 8b 84 24 20 04 00 	mov    0x420(%rsp),%rax
   6d0d2:	00 
   6d0d3:	c5 fe 6f 68 02       	vmovdqu 0x2(%rax),%ymm5
   6d0d8:	c5 fd 7f af 80 00 00 	vmovdqa %ymm5,0x80(%rdi)
   6d0df:	00 
   6d0e0:	83 fe 05             	cmp    $0x5,%esi
   6d0e3:	0f 84 87 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d0e9:	48 8b 84 24 f0 03 00 	mov    0x3f0(%rsp),%rax
   6d0f0:	00 
   6d0f1:	c5 fe 6f 68 02       	vmovdqu 0x2(%rax),%ymm5
   6d0f6:	c5 fd 7f af a0 00 00 	vmovdqa %ymm5,0xa0(%rdi)
   6d0fd:	00 
   6d0fe:	83 fe 06             	cmp    $0x6,%esi
   6d101:	0f 84 69 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d107:	8b 84 24 28 04 00 00 	mov    0x428(%rsp),%eax
   6d10e:	8b 8c 24 6c 05 00 00 	mov    0x56c(%rsp),%ecx
   6d115:	01 c8                	add    %ecx,%eax
   6d117:	01 c0                	add    %eax,%eax
   6d119:	48 98                	cltq
   6d11b:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d11f:	c4 c1 7e 6f 6c 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm5
   6d126:	c5 fd 7f af c0 00 00 	vmovdqa %ymm5,0xc0(%rdi)
   6d12d:	00 
   6d12e:	83 fe 07             	cmp    $0x7,%esi
   6d131:	0f 84 39 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d137:	44 8d 24 cd 00 00 00 	lea    0x0(,%rcx,8),%r12d
   6d13e:	00 
   6d13f:	44 89 e0             	mov    %r12d,%eax
   6d142:	29 c8                	sub    %ecx,%eax
   6d144:	48 98                	cltq
   6d146:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d14a:	c4 c1 7e 6f 64 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm4
   6d151:	c5 fd 7f a7 e0 00 00 	vmovdqa %ymm4,0xe0(%rdi)
   6d158:	00 
   6d159:	83 fe 08             	cmp    $0x8,%esi
   6d15c:	0f 84 0e 01 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d162:	c5 f8 77             	vzeroupper
   6d165:	48 8d 3d 1c dd 0e 00 	lea    0xedd1c(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6d16c:	e8 cf 7d fa ff       	call   14f40 <__tls_get_addr@plt>
   6d171:	8b b4 24 00 05 00 00 	mov    0x500(%rsp),%esi
   6d178:	48 8d b8 00 b8 00 00 	lea    0xb800(%rax),%rdi
   6d17f:	49 63 c4             	movslq %r12d,%rax
   6d182:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d186:	c4 c1 7e 6f 64 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm4
   6d18d:	c5 fd 7f a7 00 01 00 	vmovdqa %ymm4,0x100(%rdi)
   6d194:	00 
   6d195:	83 fe 09             	cmp    $0x9,%esi
   6d198:	0f 84 d2 00 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d19e:	8b 8c 24 6c 05 00 00 	mov    0x56c(%rsp),%ecx
   6d1a5:	41 8d 04 0c          	lea    (%r12,%rcx,1),%eax
   6d1a9:	48 98                	cltq
   6d1ab:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d1af:	c4 c1 7e 6f 6c 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm5
   6d1b6:	c5 fd 7f af 20 01 00 	vmovdqa %ymm5,0x120(%rdi)
   6d1bd:	00 
   6d1be:	83 fe 0a             	cmp    $0xa,%esi
   6d1c1:	0f 84 a9 00 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d1c7:	8b 94 24 e8 03 00 00 	mov    0x3e8(%rsp),%edx
   6d1ce:	8d 04 0a             	lea    (%rdx,%rcx,1),%eax
   6d1d1:	01 c0                	add    %eax,%eax
   6d1d3:	4c 63 c0             	movslq %eax,%r8
   6d1d6:	4d 6b c0 22          	imul   $0x22,%r8,%r8
   6d1da:	c4 81 7e 6f 7c 07 02 	vmovdqu 0x2(%r15,%r8,1),%ymm7
   6d1e1:	c5 fd 7f bf 40 01 00 	vmovdqa %ymm7,0x140(%rdi)
   6d1e8:	00 
   6d1e9:	83 fe 0b             	cmp    $0xb,%esi
   6d1ec:	0f 84 7e 00 00 00    	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d1f2:	01 c8                	add    %ecx,%eax
   6d1f4:	48 98                	cltq
   6d1f6:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d1fa:	c4 c1 7e 6f 74 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm6
   6d201:	c5 fd 7f b7 60 01 00 	vmovdqa %ymm6,0x160(%rdi)
   6d208:	00 
   6d209:	83 fe 0c             	cmp    $0xc,%esi
   6d20c:	74 62                	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d20e:	8b 84 24 28 04 00 00 	mov    0x428(%rsp),%eax
   6d215:	01 c8                	add    %ecx,%eax
   6d217:	c1 e0 02             	shl    $0x2,%eax
   6d21a:	4c 63 c0             	movslq %eax,%r8
   6d21d:	4d 6b c0 22          	imul   $0x22,%r8,%r8
   6d221:	c4 81 7e 6f 6c 07 02 	vmovdqu 0x2(%r15,%r8,1),%ymm5
   6d228:	c5 fd 7f af 80 01 00 	vmovdqa %ymm5,0x180(%rdi)
   6d22f:	00 
   6d230:	83 fe 0d             	cmp    $0xd,%esi
   6d233:	74 3b                	je     6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d235:	01 c8                	add    %ecx,%eax
   6d237:	48 98                	cltq
   6d239:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d23d:	c4 c1 7e 6f 6c 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm5
   6d244:	c5 fd 7f af a0 01 00 	vmovdqa %ymm5,0x1a0(%rdi)
   6d24b:	00 
   6d24c:	83 fe 0f             	cmp    $0xf,%esi
   6d24f:	75 1f                	jne    6d270 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a0>
   6d251:	6b c1 0e             	imul   $0xe,%ecx,%eax
   6d254:	48 98                	cltq
   6d256:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d25a:	c4 c1 7e 6f 6c 07 02 	vmovdqu 0x2(%r15,%rax,1),%ymm5
   6d261:	c5 fd 7f af c0 01 00 	vmovdqa %ymm5,0x1c0(%rdi)
   6d268:	00 
   6d269:	c5 f8 77             	vzeroupper
   6d26c:	eb 05                	jmp    6d273 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x9a3>
   6d26e:	66 90                	xchg   %ax,%ax
   6d270:	c5 f8 77             	vzeroupper
   6d273:	48 8d 3d 0e dc 0e 00 	lea    0xedc0e(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6d27a:	e8 c1 7c fa ff       	call   14f40 <__tls_get_addr@plt>
   6d27f:	bf 20 00 00 00       	mov    $0x20,%edi
   6d284:	48 05 00 b8 00 00    	add    $0xb800,%rax
   6d28a:	c4 e2 7b 4b 14 38    	tileloadd (%rax,%rdi,1),%tmm2
   6d290:	e9 3c f8 ff ff       	jmp    6cad1 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0x201>
   6d295:	0f 1f 00             	nopl   (%rax)
   6d298:	45 85 e4             	test   %r12d,%r12d
   6d29b:	0f 8e 92 02 00 00    	jle    6d533 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc63>
   6d2a1:	48 8d 3d e0 db 0e 00 	lea    0xedbe0(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6d2a8:	e8 93 7c fa ff       	call   14f40 <__tls_get_addr@plt>
   6d2ad:	c4 c1 7e 6f 76 02    	vmovdqu 0x2(%r14),%ymm6
   6d2b3:	48 8d 90 00 b8 00 00 	lea    0xb800(%rax),%rdx
   6d2ba:	c5 fd 7f b0 00 b8 00 	vmovdqa %ymm6,0xb800(%rax)
   6d2c1:	00 
   6d2c2:	41 83 fc 01          	cmp    $0x1,%r12d
   6d2c6:	0f 84 64 02 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d2cc:	48 8b 84 24 c0 03 00 	mov    0x3c0(%rsp),%rax
   6d2d3:	00 
   6d2d4:	c4 a1 7e 6f 64 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm4
   6d2db:	c5 fd 7f a4 24 00 05 	vmovdqa %ymm4,0x500(%rsp)
   6d2e2:	00 00 
   6d2e4:	c5 fd 7f 62 20       	vmovdqa %ymm4,0x20(%rdx)
   6d2e9:	41 83 fc 02          	cmp    $0x2,%r12d
   6d2ed:	0f 84 3d 02 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d2f3:	48 8b 84 24 b8 03 00 	mov    0x3b8(%rsp),%rax
   6d2fa:	00 
   6d2fb:	c4 a1 7e 6f 64 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm4
   6d302:	c5 fd 7f a4 24 00 05 	vmovdqa %ymm4,0x500(%rsp)
   6d309:	00 00 
   6d30b:	c5 fd 7f 62 40       	vmovdqa %ymm4,0x40(%rdx)
   6d310:	41 83 fc 03          	cmp    $0x3,%r12d
   6d314:	0f 84 16 02 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d31a:	48 8b 84 24 a8 03 00 	mov    0x3a8(%rsp),%rax
   6d321:	00 
   6d322:	c4 a1 7e 6f 74 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm6
   6d329:	c5 fd 7f b4 24 00 05 	vmovdqa %ymm6,0x500(%rsp)
   6d330:	00 00 
   6d332:	c5 fd 7f 72 60       	vmovdqa %ymm6,0x60(%rdx)
   6d337:	41 83 fc 04          	cmp    $0x4,%r12d
   6d33b:	0f 84 ef 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d341:	48 8b 84 24 98 03 00 	mov    0x398(%rsp),%rax
   6d348:	00 
   6d349:	c4 a1 7e 6f 74 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm6
   6d350:	c5 fd 7f b4 24 00 05 	vmovdqa %ymm6,0x500(%rsp)
   6d357:	00 00 
   6d359:	c5 fd 7f b2 80 00 00 	vmovdqa %ymm6,0x80(%rdx)
   6d360:	00 
   6d361:	41 83 fc 05          	cmp    $0x5,%r12d
   6d365:	0f 84 c5 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d36b:	48 8b 84 24 90 03 00 	mov    0x390(%rsp),%rax
   6d372:	00 
   6d373:	c4 a1 7e 6f 6c 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm5
   6d37a:	c5 fd 7f ac 24 00 05 	vmovdqa %ymm5,0x500(%rsp)
   6d381:	00 00 
   6d383:	c5 fd 7f aa a0 00 00 	vmovdqa %ymm5,0xa0(%rdx)
   6d38a:	00 
   6d38b:	41 83 fc 06          	cmp    $0x6,%r12d
   6d38f:	0f 84 9b 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d395:	48 8b 84 24 88 03 00 	mov    0x388(%rsp),%rax
   6d39c:	00 
   6d39d:	c4 a1 7e 6f 74 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm6
   6d3a4:	c5 fd 7f b4 24 00 05 	vmovdqa %ymm6,0x500(%rsp)
   6d3ab:	00 00 
   6d3ad:	c5 fd 7f b2 c0 00 00 	vmovdqa %ymm6,0xc0(%rdx)
   6d3b4:	00 
   6d3b5:	41 83 fc 07          	cmp    $0x7,%r12d
   6d3b9:	0f 84 71 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d3bf:	48 8b 84 24 80 03 00 	mov    0x380(%rsp),%rax
   6d3c6:	00 
   6d3c7:	c4 a1 7e 6f 64 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm4
   6d3ce:	c5 fd 7f a4 24 00 05 	vmovdqa %ymm4,0x500(%rsp)
   6d3d5:	00 00 
   6d3d7:	c5 fd 7f a2 e0 00 00 	vmovdqa %ymm4,0xe0(%rdx)
   6d3de:	00 
   6d3df:	41 83 fc 08          	cmp    $0x8,%r12d
   6d3e3:	0f 84 47 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d3e9:	c5 f8 77             	vzeroupper
   6d3ec:	48 8d 3d 95 da 0e 00 	lea    0xeda95(%rip),%rdi        # 15ae88 <_DYNAMIC+0x240>
   6d3f3:	e8 48 7b fa ff       	call   14f40 <__tls_get_addr@plt>
   6d3f8:	48 8d 90 00 b8 00 00 	lea    0xb800(%rax),%rdx
   6d3ff:	48 8b 84 24 78 03 00 	mov    0x378(%rsp),%rax
   6d406:	00 
   6d407:	c4 a1 7e 6f 6c 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm5
   6d40e:	c5 fd 7f ac 24 00 05 	vmovdqa %ymm5,0x500(%rsp)
   6d415:	00 00 
   6d417:	c5 fd 7f aa 00 01 00 	vmovdqa %ymm5,0x100(%rdx)
   6d41e:	00 
   6d41f:	41 83 fc 09          	cmp    $0x9,%r12d
   6d423:	0f 84 07 01 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d429:	48 8b 84 24 68 03 00 	mov    0x368(%rsp),%rax
   6d430:	00 
   6d431:	c4 a1 7e 6f 6c 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm5
   6d438:	c5 fd 7f ac 24 00 05 	vmovdqa %ymm5,0x500(%rsp)
   6d43f:	00 00 
   6d441:	c5 fd 7f aa 20 01 00 	vmovdqa %ymm5,0x120(%rdx)
   6d448:	00 
   6d449:	41 83 fc 0a          	cmp    $0xa,%r12d
   6d44d:	0f 84 dd 00 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d453:	48 8b 84 24 50 03 00 	mov    0x350(%rsp),%rax
   6d45a:	00 
   6d45b:	c4 a1 7e 6f 74 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm6
   6d462:	c5 fd 7f b4 24 00 05 	vmovdqa %ymm6,0x500(%rsp)
   6d469:	00 00 
   6d46b:	c5 fd 7f b2 40 01 00 	vmovdqa %ymm6,0x140(%rdx)
   6d472:	00 
   6d473:	41 83 fc 0b          	cmp    $0xb,%r12d
   6d477:	0f 84 b3 00 00 00    	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d47d:	8b bc 24 6c 05 00 00 	mov    0x56c(%rsp),%edi
   6d484:	8b 84 24 e8 03 00 00 	mov    0x3e8(%rsp),%eax
   6d48b:	48 8b b4 24 20 04 00 	mov    0x420(%rsp),%rsi
   6d492:	00 
   6d493:	01 f8                	add    %edi,%eax
   6d495:	8d 04 47             	lea    (%rdi,%rax,2),%eax
   6d498:	48 98                	cltq
   6d49a:	48 ff c0             	inc    %rax
   6d49d:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d4a1:	48 01 f0             	add    %rsi,%rax
   6d4a4:	c4 a1 7e 6f 74 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm6
   6d4ab:	c5 fd 7f b2 60 01 00 	vmovdqa %ymm6,0x160(%rdx)
   6d4b2:	00 
   6d4b3:	41 83 fc 0c          	cmp    $0xc,%r12d
   6d4b7:	74 77                	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d4b9:	8b 84 24 28 04 00 00 	mov    0x428(%rsp),%eax
   6d4c0:	01 f8                	add    %edi,%eax
   6d4c2:	c1 e0 02             	shl    $0x2,%eax
   6d4c5:	48 63 c8             	movslq %eax,%rcx
   6d4c8:	48 ff c1             	inc    %rcx
   6d4cb:	48 6b c9 22          	imul   $0x22,%rcx,%rcx
   6d4cf:	48 01 f1             	add    %rsi,%rcx
   6d4d2:	c4 a1 7e 6f 7c 29 02 	vmovdqu 0x2(%rcx,%r13,1),%ymm7
   6d4d9:	c5 fd 7f ba 80 01 00 	vmovdqa %ymm7,0x180(%rdx)
   6d4e0:	00 
   6d4e1:	41 83 fc 0d          	cmp    $0xd,%r12d
   6d4e5:	74 49                	je     6d530 <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xc60>
   6d4e7:	01 f8                	add    %edi,%eax
   6d4e9:	48 98                	cltq
   6d4eb:	48 ff c0             	inc    %rax
   6d4ee:	48 6b c0 22          	imul   $0x22,%rax,%rax
   6d4f2:	48 01 f0             	add    %rsi,%rax
   6d4f5:	c4 a1 7e 6f 7c 28 02 	vmovdqu 0x2(%rax,%r13,1),%ymm7
   6d4fc:	c5                   	.byte 0xc5
   6d4fd:	fd                   	std
   6d4fe:	7f ba                	jg     6d4ba <void (anonymous namespace)::tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>(int, int, int, void const*, void const*, float*, int) [clone .constprop.0]+0xbea>
