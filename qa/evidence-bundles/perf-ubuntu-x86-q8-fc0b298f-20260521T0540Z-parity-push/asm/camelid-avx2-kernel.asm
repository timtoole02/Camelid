
target/release/camelid:     file format elf64-x86-64


Disassembly of section .text:

000000000034e270 <camelid::inference::q8_0_packed_4x8_block_avx2>:
  34e270:	c5 fe 6f 06          	vmovdqu (%rsi),%ymm0
  34e274:	c5 fe 6f 4e 20       	vmovdqu 0x20(%rsi),%ymm1
  34e279:	c5 fe 6f 56 40       	vmovdqu 0x40(%rsi),%ymm2
  34e27e:	c5 fe 6f 5e 60       	vmovdqu 0x60(%rsi),%ymm3
  34e283:	c4 e2 7d 08 e0       	vpsignb %ymm0,%ymm0,%ymm4
  34e288:	c4 e2 7d 59 2a       	vpbroadcastq (%rdx),%ymm5
  34e28d:	c4 e2 55 08 c0       	vpsignb %ymm0,%ymm5,%ymm0
  34e292:	c4 e2 5d 04 c0       	vpmaddubsw %ymm0,%ymm4,%ymm0
  34e297:	c4 e2 7d 79 25 a2 fd 	vpbroadcastw -0x2f025e(%rip),%ymm4        # 5e042 <anon.e69a9f855bc12f31b5eeb0a1baedb7ba.432.llvm.15554514708717879527+0x568c>
  34e29e:	d0 ff
  34e2a0:	c5 fd f5 c4          	vpmaddwd %ymm4,%ymm0,%ymm0
  34e2a4:	c4 e2 7d 59 6a 08    	vpbroadcastq 0x8(%rdx),%ymm5
  34e2aa:	c4 e2 75 08 f1       	vpsignb %ymm1,%ymm1,%ymm6
  34e2af:	c4 e2 55 08 c9       	vpsignb %ymm1,%ymm5,%ymm1
  34e2b4:	c4 e2 4d 04 c9       	vpmaddubsw %ymm1,%ymm6,%ymm1
  34e2b9:	c5 f5 f5 cc          	vpmaddwd %ymm4,%ymm1,%ymm1
  34e2bd:	c5 f5 fe c0          	vpaddd %ymm0,%ymm1,%ymm0
  34e2c1:	c4 e2 6d 08 ca       	vpsignb %ymm2,%ymm2,%ymm1
  34e2c6:	c4 e2 7d 59 6a 10    	vpbroadcastq 0x10(%rdx),%ymm5
  34e2cc:	c4 e2 55 08 d2       	vpsignb %ymm2,%ymm5,%ymm2
  34e2d1:	c4 e2 75 04 ca       	vpmaddubsw %ymm2,%ymm1,%ymm1
  34e2d6:	c5 f5 f5 cc          	vpmaddwd %ymm4,%ymm1,%ymm1
  34e2da:	c4 e2 65 08 d3       	vpsignb %ymm3,%ymm3,%ymm2
  34e2df:	c4 e2 7d 59 6a 18    	vpbroadcastq 0x18(%rdx),%ymm5
  34e2e5:	c4 e2 55 08 db       	vpsignb %ymm3,%ymm5,%ymm3
  34e2ea:	c4 e2 6d 04 d3       	vpmaddubsw %ymm3,%ymm2,%ymm2
  34e2ef:	c5 ed f5 d4          	vpmaddwd %ymm4,%ymm2,%ymm2
  34e2f3:	c5 f5 fe ca          	vpaddd %ymm2,%ymm1,%ymm1
  34e2f7:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  34e2fb:	c4 e3 7d 39 c1 01    	vextracti128 $0x1,%ymm0,%xmm1
  34e301:	c4 e2 79 02 c1       	vphaddd %xmm1,%xmm0,%xmm0
  34e306:	c5 fa 7f 07          	vmovdqu %xmm0,(%rdi)
  34e30a:	c5 f8 77             	vzeroupper
  34e30d:	c3                   	ret
  34e30e:	cc                   	int3
  34e30f:	cc                   	int3

000000000034e310 <camelid::inference::q8_0_selected_packed_rows4>:
  34e310:	41 56                	push   %r14
  34e312:	53                   	push   %rbx
  34e313:	50                   	push   %rax
  34e314:	48 89 fb             	mov    %rdi,%rbx
  34e317:	31 c0                	xor    %eax,%eax
  34e319:	48 3b 87 40 01 00 00 	cmp    0x140(%rdi),%rax
  34e320:	71 33                	jno    34e355 <camelid::inference::q8_0_selected_packed_rows4+0x45>
  34e322:	49 be 00 00 00 00 00 	movabs $0x8000000000000000,%r14
  34e329:	00 00 80
  34e32c:	48 8d 3d 0c 29 d1 ff 	lea    -0x2ed6f4(%rip),%rdi        # 60c3f <anon.279777848e1acd0ef405bca26607e370.561.llvm.10202336670292593932+0xaa9>
  34e333:	be 1b 00 00 00       	mov    $0x1b,%esi
  34e338:	e8 e3 1d fd ff       	call   320120 <camelid::inference::env_flag_enabled>
  34e33d:	84 c0                	test   %al,%al
  34e33f:	74 2d                	je     34e36e <camelid::inference::q8_0_selected_packed_rows4+0x5e>
  34e341:	4c 39 b3 d0 00 00 00 	cmp    %r14,0xd0(%rbx)
  34e348:	74 24                	je     34e36e <camelid::inference::q8_0_selected_packed_rows4+0x5e>
  34e34a:	48 81 c3 d0 00 00 00 	add    $0xd0,%rbx
  34e351:	b2 01                	mov    $0x1,%dl
  34e353:	eb 0e                	jmp    34e363 <camelid::inference::q8_0_selected_packed_rows4+0x53>
  34e355:	0f b6 93 a8 01 00 00 	movzbl 0x1a8(%rbx),%edx
  34e35c:	48 81 c3 40 01 00 00 	add    $0x140,%rbx
  34e363:	48 89 d8             	mov    %rbx,%rax
  34e366:	48 83 c4 08          	add    $0x8,%rsp
  34e36a:	5b                   	pop    %rbx
  34e36b:	41 5e                	pop    %r14
  34e36d:	c3                   	ret
  34e36e:	48 8d 3d af 28 d1 ff 	lea    -0x2ed751(%rip),%rdi        # 60c24 <anon.279777848e1acd0ef405bca26607e370.561.llvm.10202336670292593932+0xa8e>
  34e375:	be 1b 00 00 00       	mov    $0x1b,%esi
  34e37a:	e8 a1 1d fd ff       	call   320120 <camelid::inference::env_flag_enabled>
  34e37f:	84 c0                	test   %al,%al
  34e381:	74 0f                	je     34e392 <camelid::inference::q8_0_selected_packed_rows4+0x82>
  34e383:	4c 39 73 60          	cmp    %r14,0x60(%rbx)
  34e387:	48 8d 5b 60          	lea    0x60(%rbx),%rbx
  34e38b:	0f 94 c2             	sete   %dl
  34e38e:	00 d2                	add    %dl,%dl
  34e390:	eb d1                	jmp    34e363 <camelid::inference::q8_0_selected_packed_rows4+0x53>
  34e392:	b2 02                	mov    $0x2,%dl
  34e394:	eb cd                	jmp    34e363 <camelid::inference::q8_0_selected_packed_rows4+0x53>
  34e396:	cc                   	int3
  34e397:	cc                   	int3
  34e398:	cc                   	int3
  34e399:	cc                   	int3
  34e39a:	cc                   	int3
  34e39b:	cc                   	int3
  34e39c:	cc                   	int3
  34e39d:	cc                   	int3
  34e39e:	cc                   	int3
  34e39f:	cc                   	int3

000000000034e3a0 <camelid::inference::tensor_window_around_index>:
  34e3a0:	55                   	push   %rbp
  34e3a1:	41 57                	push   %r15
  34e3a3:	41 56                	push   %r14
  34e3a5:	41 55                	push   %r13
  34e3a7:	41 54                	push   %r12
  34e3a9:	53                   	push   %rbx
  34e3aa:	50                   	push   %rax
  34e3ab:	45 31 f6             	xor    %r14d,%r14d
  34e3ae:	48 83 e9 05          	sub    $0x5,%rcx
  34e3b2:	4c 0f 43 f1          	cmovae %rcx,%r14
  34e3b6:	4d 8d 6e 0a          	lea    0xa(%r14),%r13
  34e3ba:	49 39 d5             	cmp    %rdx,%r13
  34e3bd:	4c 0f 43 ea          	cmovae %rdx,%r13
  34e3c1:	4d 89 ec             	mov    %r13,%r12
  34e3c4:	4d 29 f4             	sub    %r14,%r12
  34e3c7:	72 6b                	jb     34e434 <camelid::inference::tensor_window_around_index+0x94>
  34e3c9:	48 89 fb             	mov    %rdi,%rbx
  34e3cc:	49 c1 e4 02          	shl    $0x2,%r12
  34e3d0:	4d 89 ef             	mov    %r13,%r15
  34e3d3:	bd 04 00 00 00       	mov    $0x4,%ebp
  34e3d8:	4d 29 f7             	sub    %r14,%r15
  34e3db:	74 24                	je     34e401 <camelid::inference::tensor_window_around_index+0x61>
  34e3dd:	48 89 34 24          	mov    %rsi,(%rsp)
  34e3e1:	ff 15 f9 e9 33 00    	call   *0x33e9f9(%rip)        # 68cde0 <_DYNAMIC+0x240>
  34e3e7:	be 04 00 00 00       	mov    $0x4,%esi
  34e3ec:	4c 89 e7             	mov    %r12,%rdi
  34e3ef:	ff 15 f3 e9 33 00    	call   *0x33e9f3(%rip)        # 68cde8 <_DYNAMIC+0x248>
  34e3f5:	48 85 c0             	test   %rax,%rax
  34e3f8:	74 4d                	je     34e447 <camelid::inference::tensor_window_around_index+0xa7>
  34e3fa:	48 89 c5             	mov    %rax,%rbp
  34e3fd:	48 8b 34 24          	mov    (%rsp),%rsi
  34e401:	4d 39 f5             	cmp    %r14,%r13
  34e404:	74 10                	je     34e416 <camelid::inference::tensor_window_around_index+0x76>
  34e406:	4a 8d 34 b6          	lea    (%rsi,%r14,4),%rsi
  34e40a:	48 89 ef             	mov    %rbp,%rdi
  34e40d:	4c 89 e2             	mov    %r12,%rdx
  34e410:	ff 15 f2 e9 33 00    	call   *0x33e9f2(%rip)        # 68ce08 <memcpy@GLIBC_2.14>
  34e416:	4c 89 33             	mov    %r14,(%rbx)
  34e419:	4c 89 7b 08          	mov    %r15,0x8(%rbx)
  34e41d:	48 89 6b 10          	mov    %rbp,0x10(%rbx)
  34e421:	4c 89 7b 18          	mov    %r15,0x18(%rbx)
  34e425:	48 83 c4 08          	add    $0x8,%rsp
  34e429:	5b                   	pop    %rbx
  34e42a:	41 5c                	pop    %r12
  34e42c:	41 5d                	pop    %r13
  34e42e:	41 5e                	pop    %r14
