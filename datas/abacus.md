# abacus.pdf

> MinerU Flash 批处理解析  |  共 362 页  |  19 个切片

---

## 第 1–20 页

# O O O ABACUS

原 子 一算 筹

# 原子算筹（ABACUS）软件使用和开发手册(V0.1版本）

作者：ABACUS 爱好者

组织：北京大学

时间：2025年8月6日

## 目录

第1章基本介绍 1  
1.1前言．.  
1.2相关网址 1  
1.3交流平台 1  
1.4常见问题 1  
第2章编译ABACUS 6  
2.1GCC 编译 ABACUS 教程 6  
2.1.1介绍：关于ABACUS 6  
2.1.2 检查当前服务器基础环境： 6  
2.1.3安装仅支持PW 基组的 ABACUS 7  
2.1.4 安装支持两种基组的ABACUS. 8  
2.2Intel oneAPI 2024/2025 编译 ABACUS 教程 9  
2.2.1Intel OneAPI工具链．. 9  
2.2.2 ABACUS 安装 12  
2.2.3 常见问题和解决方法 15  
2.2.4参考 17  
2.3Intel oneAPI编译 ABACUS 教程 18  
2.3.1前言 18  
2.3.2关于Intel@ oneAPITolkits. 18  
2.3.3编译安装 18  
2.4Toolchain 编译 ABACUS 教程(1)-GCC 20  
2.4.1简介 20  
2.4.2什么是 ABACUS Toolchain. 20  
2.4.3基于 GCC Toolchain 安装 ABACUS 21  
2.5Toolchain 编译 ABACUS 教程(2)-Intel 29  
2.5.1简介 29  
2.5.2基于Intel Toolchain 安装 ABACUS. 29  
2.5.3基于 Intel Toolchain 引入 ABACUS 功能插件 42  
2.6Toolchain 编译 ABACUS 教程(3)-AMD 44  
2.6.1简介 44  
2.6.2已有方案回顾 45  
2.6.3AMD Toolchain 45  
2.7编译 Nvidia GPU 版本的 ABACUS 54  
2.7.1介绍 54  
2.7.2检查环境 55  
2.7.3编译 55  
2.8ABACUS LCAO 基组GPU 版本使用说明 55  
2.8.1简介 55  
2.8.2安装 55  
2.8.3Toolchain 安装 56  
2.8.4使用 56  
2.8.5示例 57  
2.9在超算环境编译 ABACUS 的建议 65  
2.9.1关于 ABACUS 65  
2.9.2 超算平台的简单介绍 65  
2.9.3 拓展教程 67  
2.10 ABACUS 在曙光DCU 集群与 AMD 显卡上的编译与使用 67  
2.10.1介绍 67  
2.10.2准备 67  
2.10.3流程 68  
2.10.4结语 71  
3章建模 72  
3.1ABACUS 结构文件 STRU及其准备方式. 72  
3.1.1 背景 72  
3.1.2 基于 ASE-ABACUS 快速准备 STRU 文件 72  
3.1.3 STRU 文件的基本结构 73  
3.1.4 基于 ASE-ABACUS 从其他结构文件准备 STRU 74  
3.1.5 基于ATOMKIT准备 STRU 文件 75  
3.1.6 将 STRU 文件转化为其他格式. 78  
3.1.7 基于 abacustest 准备 ABACUS 输入文件 78  
3.2 ABACUS 如何选择晶胞朝向获得最佳并行效率？以碳纳米管为例 79  
3.2.1 背景 79  
3.2.2 ABACUS做不同真空方向计算的流程. 79  
3.2.3 测试结果 81  
3.3ABACUS 如何选择晶胞朝向获得最佳并行效率？以二维氮化硼为例 82  
3.3.1 背景 82  
3.3.2ABACUS 计算的流程 82  
3.3.3 测试结果 89  
3.4ABACUS 如何选择晶胞朝向获得最佳并行效率？以铜表面一氧化碳吸附为例 89  
3.4.1 背景 89  
3.4.2ABACUS 做不同真空方向计算的流程. 90  
3.4.3 测试结果 98  
3.5模守恒赝势生成方法简介 99  
3.5.1 介绍 99  
3.5.2 从其他格式赝势转换 99  
3.5.3 赝势的生成 99  
3.5.4 结尾 110  
3.6数值原子轨道（一)：ABACUS中的数值原子轨道命名和使用方法 110  
3.6.1 数值原子轨道的背景知识 110  
3.6.2 数值原子轨道的命名方法 111  
3.6.3 ABACUS 中数值原子轨道的使用方法. 112  
3.6.4参考文献．． 113  
3.7 数值原子轨道(二)：生成给定模守恒赝势的数值原子轨道 113  
3.7.1ABACUS 中的数值原子轨道背景知识. 113  
3.7.2ABACUS 中产生数值原子轨道的具体流程 116  
3.7.3ABACUS 中数值原子轨道的生成脚本， 120  
3.7.4ABACUS 中数值原子轨道的测试流程 123  
3.7.5参考文献． 124  
3.8 数值原子轨道(三)：产生高精度数值原子轨道 124  
3.8.1 PTG（PyTorch Gradient）方法 124  
3.8.2 PTG_dpsi（PyTorch Gradient with dpsi）方法 124  
3.8.3 安装 Pytorch 125  
3.8.4 产生高精度数值原子轨道流程 125  
3.8.5 参考文献 127  
;4章基础功能 128  
4.1 Kohn-Sham 密度泛函理论计算 128  
4.1.1平面波计算与收敛性测试 128  
4.1.2ABACUS+LibRI做杂化泛函计算教程 136  
4.1.3ABACUS 收敛性问题解决手册． 140  
4.2分子动力学模拟 144  
4.2.1ABACUS 分子动力学使用教程. 144  
4.2.2ABACUS+DeePMD-kit 做机器学习分子动力学 154  
4.3采用ABACUS 进行表面计算（一)：静电势和功函数 158  
4.3.1 背景 158  
4.3.2 ABACUS 中计算静电势具体流程 158  
4.4采用 ABACUS 进行表面计算（二)：偶极修正 161  
4.4.1 偶极修正介绍（Dipole Correction） 161  
4.4.2 文档与算例 162  
4.4.3 偶极修正输人参数介绍 162  
4.4.4 预期结果 164  
4.4.5 参考文献 164  
4.5采用 ABACUS 进行表面计算（三)：表面能计算 165  
4.5.1 背景 165  
4.5.2ABACUS 计算表面能具体流程， 165  
4.5.3参考文献． 167  
4.6采用ABACUS 进行表面计算（四)：表面缺陷能和吸附能计算 167  
4.6.1 背景 167  
4.6.2 ABACUS 计算表面缺陷能具体流程 168  
4.6.3 ABACUS 计算表面吸附能具体流程 169  
4.6.4 参考文献 171  
4.7 采用 ABACUS 进行表面计算（五)：外加电场 171  
4.7.1 介绍 171  
4.7.2 软件和算例准备 172  
4.7.3 采用加电场进行二维材料的能带计算 172  
4.7.4 预期计算结果 174  
4.7.5 参考文献 176  
4.8采用 ABACUS 进行表面计算（六)：补偿电荷 176  
4.8.1介绍 176  
4.8.2文档与算例 177  
4.8.3 偶极修正与补偿电荷结合的表面计算 177  
4.8.4预期结果 178  
4.8.5参考文献 179  
第5章特色功能 180  
5.1ABACUS 随机波函数DFT方法使用教程 180  
5.1.1 介绍 180  
5.1.2 软件和算例准备： 180  
5.1.3 采用 SDFT进行电子自洽迭代计算 180  
5.1.4 采用 SDFT进行分子动力学模拟. 182  
5.1.5 采用 SDFT 计算态密度 182  
5.1.6结尾 183  
5.2 ABACUS 随机波函数DFT计算电子电导热导教程 184  
5.2.1介绍 184  
5.2.2算例准备 184  
5.2.3采用 sDFT进行电子的电导热导计算 185  
5.3 ABACUS实时演化含时密度泛函理论使用教程 187  
5.3.1 背景 187  
5.3.2在 ABACUS 中使用rt-TDDFT时改变电子的占据数. 187  
5.3.3ABACUS 中使用rt-TDDFT 外加电场计算材料吸收谱 190  
5.3.4 参考文献 193  
5.4 ABACUS无轨道密度泛函理论方法使用教程 193  
5.4.1无轨道密度泛函理论背景知识． 193  
5.4.2ABACUS 中进行 OFDFT 计算的具体流程 195  
5.4.3 参考文献 198  
5.5ABACUS 隐式溶剂模型使用教程 198  
5.5.1 介绍 198  
5.5.2 文档和算例 199  
5.5.3 输人参数 199  
5.5.4 预期结果 200  
5.5.5 参考文献 201  
5.6 ABACUS+DPGEN 使用教程 201  
5.6.1 介绍 201  
5.6.2准备 202  
5.6.3结合DPGEN 生成 DP数据的流程 202  
5.6.4计算材料 EOS 和弹性常数的流程 212  
第6章软件接口与后处理 215  
6.1ABACUS+Atomkit 计算态密度和能带 215  
6.1.1 介绍 215  
6.1.2 准备 215  
6.1.3 铝的态密度以及电子能带结构计算 216  
6.1.4铁的态密度以及电子能带结构计算 221  
6.1.5 参考文献 222  
6.2ABACUS 计算 PDOS 223  
6.2.1 PDOS 文件分析 223  
6.2.2绘制 PDOS 224  
6.2.3绘制 PDOS 的流程 227  
6.3ABACUS 输出部分的电荷密度和波函数及可视化教程 228  
6.3.1 引言 228  
6.3.2 基本概念与公式 228  
6.3.3 输入输出概述 229  
6.3.4 在平面波基组下进行计算 231  
6.3.5 在数值原子轨道基组下进行计算 232  
6.3.6 示例：对电荷密度进行可视化 234  
6.3.7 结语 236  
6.4ABACUS 计算电子局域函数 ELF 使用教程 237  
6.4.1 背景 237  
6.4.2 定义 237  
6.4.3 ABACUS 计算ELF具体流程. 238  
6.4.4 计算实例：自旋极化的体心立方铁 240  
6.4.5 参考文献 241  
6.5ABACUS+Bader charge 分析教程 241  
6.5.1 前言 241  
6.5.2 Bader 分析软件部署. 241  
6.5.3 使用示例 242  
6.5.4 附录：cube manipulator 更多的使用方法 245  
6.5.5Have a question? Submit issue! 250  
6.6ABACUS+pymatgen 计算弹性常数 250  
6.6.1介绍 250  
6.6.2准备 251  
6.6.3金刚石的弹性常数计算 252  
6.7ABACUS+Phonopy 计算声子谱 253  
6.7.1介绍 253  
6.7.2准备 253  
6.7.3流程 254  
6.8ABACUS+ShengBTE 计算晶格热导率 258  
6.8.1介绍 258  
6.8.2准备 258  
6.8.3流程 258  
6.8.4结尾 262  
6.9ABACUS+Wannier90 使用教程 262  
6.9.1 教程使用说明 262  
6.9.2Wannier90软件介绍. 262  
6.9.3 使用 ABACUS 准备Wannier90 需要的文件 262  
6.9.4 参考文献． 265  
6.10 ABACUS+Candela 使用教程 265  
6.10.1介绍 265  
6.10.2准备 265  
6.10.3流程 266  
6.11 ABACUS+USPEX 接口教程 270  
6.11.1 背景介绍 270  
6.11.2接口介绍 270  
6.11.3流程(以 Si为例) 273  
6.11.4结语 276  
6.12 ABACUS+Hefei NAMD 使用教程 277  
6.12.1介绍 277  
6.12.2 计算流程 277  
6.12.3结语 280  
6.12.4 参考文献． 280  
6.13 Pyabacus 文档一：用户手册 280  
6.13.1简介 280  
6.13.2模块 280  
6.13.3安装 281  
6.13.4调用 281  
6.13.5示例 281  
6.13.6测试 282  
6.13.7结语 282  
6.14 Pyabacus 文档二：HSolver 模块 282  
6.14.1简介 282  
6.14.2 安装 pyabacus 284  
6.14.3调用对角化算法 284  
6.14.4 结语 285  
6.15 Pyabacus 文档三：开发者指南 285  
6.15.1简介 285  
6.15.2 结构 285  
6.15.3资料汇总 290  
6.15.4结语 290  
i7章开发者规范 291  
7.1ABACUS 的 Github 仓库 Issues 处理流程 291  
7.1.1Github 的 Issues 机制 291  
7.1.2ABACUS 项目的 Issues 标签分类 292  
7.1.3Issues 处理流程 292  
7.2ABACUS 开源项目C++代码规范 299  
7.2.1 前言 299  
7.2.2 命名约定． 299  
7.2.3 关于头文件 300  
7.2.4 关于类（Class） 301  
7.2.5 关于函数（Function） 302  
7.2.6 关于作用域 303  
7.2.7 其他C++特性． 305  
7.2.8 关于 ABACUS 中常用的关键词缩写. 308  
7.2.9Reference． 308  
7.3ABACUS 注释规范：Doxygen 入门(c++) 309  
7.3.1简介 309  
7.3.2如何使用Doxygen 312  
7.3.3附录：Doxygen 与VS Code． 322  
7.4ABACUS 线上文档输入参数撰写规范 322  
7.4.1背景 322  
7.4.2规范格式. 322  
7.4.3其他问题 324  
7.5ABACUS 代码存放规范 324  
7.5.1背景 324  
7.5.2代码目录． 324  
7.5.3总结 327  
7.6如何在 ABACUS 中新增一个输人参数（截至 v3.5.3） 327  
7.6.1在 input 类中申明和定义. 327  
7.6.2将 input 成员转换为全局变量 329  
7.6.3添加文档．. 330  
7.7如何在 ABACUS 中新增一个输入参数（v3.7.0后） 330  
7.7.1开发者须知 330  
7.7.2设计文档参考 334  
7.8如何在 ABACUS 中进行异构计算 338  
7.8.1前言 338  
7.8.2 管理设备类型 338  
7.8.3数据内存管理（旧) 339  
7.8.4 数据内存管理(新) 345  
7.8.5执行数学计算 346  
7.9ABACUS formatter-2.0 版本使用说明书 347  
7.9.1前言 ，： 347  
7.9.2字符串格式化 347  
7.9.3制表功能． 350  
7.9.4static Python-style 字符串函数 351  
7.9.5Find a bug? Submitissue! 352  
7.10 ABACUS 中使用格式化工具 clang-format 352  
7.10.1 引言 352  
7.10.2 在 ABACUS 中使用 clang-format 352  
7.10.3格式化时的注意事项: 353  
7.10.4 参考文献： 354

## 第1章 基本介绍

## 1.1 前言

本文档素材主要收集自 ABACUS 线上中文文档：htps://mcresearch.github.io/abacus-user-guide/，希望能够惠及 ABACUS 软件的开发者和使用者。若有任何疑问，请写邮件到 mohanchen@pku.edu.cn进一步交流。

## 1.2 相关网址

ABACUS 的官网地址为：http://abacus.ustc.edu.cn/

ABACUS 在 DeepModeling 社区中的 GitHub 仓库地址为：https:/github.com/deepmodeling/abacus-develop

ABACUS 的 Gitee 镜像仓库地址为：https://gitee.com/deepmodeling/abacus-develop

ABACUS 线上文档（包括输入输出参数介绍等)：https://abacus.deepmodeling.com/en/latest/

ABACUS 线上中文文档：https://mcresearch.github.io/abacus-user-guide/

ABACUS 开源社区提间与讨论：https://github.com/deepmodeling/abacus-develop/issues

## 1.3 交流平台

ABACUS社区目前有微信群和QQ群。

ABACUS用户QQ群：759914681。

## 1.4 常见问题

## 常用缩写

。 PW:Plane Wave，指平面波

NAO:Numerical Atomic Orbitals，指数值原子轨道

·LCAO:Linear Combination of Atomic Orbitals，指局域轨道，在 ABACUS 里面也指数值原子轨道

常见问题基于过去几年来积累的用户问题，接下来本文档将尽量以问答的形式来替使用ABACUS 的用户答疑解惑，例如：

1.问：ABACUS 和ABAQUS 是什么关系？

答：ABAQUS 是一款知名的多物理场有限元仿真分析软件，ABACUS的英文单词翻译是算盘，ABACUS作为一款国产密度泛函理论软件，中文名是原子算筹。

2.问：使用ABACUS需要收费吗？

答：从软件的角度来讲不收费，因为是开源软件，但跑该软件的机器得自己准备，机器要交电费。

3．问：ABACUS的开发过程是怎样的，什么人可以成为开发者？

答：ABACUS 的代码开发记录完全公开，所有代码的修改都在 https://github.com/deepmodeling/abacus-develop 网页上可以看到，ABACUS 为所有感兴趣的参与者都保留了成为开发者的可能性，每个人都有可能成为开发者。

## ABACUS使用建议

## 1.编译

## (a)．从哪里可以下载到ABACUS的代码？

答：如果可以接 github，可以登录https://github.com/deepmodeling/abacus-develop，建议下载最新版本使用，因为软件一直在更新，总体而言新版本的bugs会更少。

如果网络不方便登录github，还可以登录国内网站 https://gitee.com/deepmodeling/abacus-develop.git 下载。

## (b)．怎么安装ABACUS?

答：可以参考官方文档 https://abacus.depmodeling.com/en/latest/quick_start/easy_ install.html

也可以登录网址 https://mcresearch.gitee.io/abacus-user-guide/，上面有一些针对不同编译器的中文编译教程。

另外，注意“安装 ABACUS”这句话其实有歧义，ABACUS有很多不同的安装选项，比如是否只安装支持平面波的功能，还是也要安装支持数值原子轨道的功能，是否要使用LIBXC 等等不同选项，可根据自己的需要来安装具体的代码模块。

## (c)．如何在曙光的DCU上编译ABACUS？

答:可参考文档https://mcresearch.gitee.io/abacus-user-guide/abacus-dcu.html

## 2．计算前准备

## (a).PW计算需要准备的文件？

答：INPUT文件控制输入参数，STRU文件包含晶胞、原子信息和赝势；KPT文件包含布里渊区k点信息。这三个文件默认是放在一个文件夹内的。但是 INPUT文件里其实可以控制结构文件所在地址，以及 KPT 文件所在地址。

## (b).KPT文件什么时候不适用？

答：第一，在使用 DPGEN的时候，强烈建议设置kspacing 这个参数，而不是在KPT文件里设置k点的个数，这是为了做不同大小体系密度泛函计算的时候，使得计算的精度尽量持平，从而得到不同结构里精度相当的总能量和受力，这对于DPMD 的训练很重要。第二，LCAO可以在INPUT 使用gamma_only（设为1)，那么KPT文件不起作用，所有计算只用到Gamma（0，0，0）这个k点，这种情况下体系有更快速的特殊算法，效率更高，计算内存的消耗也更小。

## (c).LCAO基组相比于PW基组需要额外准备的文件？

答：数值原子轨道文件，下载地址为https://abacus.ustc.edu.cn/pseudo/list.htm。注意轨道文件要和对应的赝势相匹配。

关于数值原子轨道的背景知识，可以参考文献：

· https://mcresearch.gitee.io/abacus-user-guide/abacus-nac1.html

· https://mcresearch.gitee.io/abacus-user-guide/abacus-nac2.html

https://mcresearch.gitee.io/abacus-user-guide/abacus-nac3.html

## (d)．能量截断值收敛性测试是必须要做的吗？

答：是的，除非你或者别人之前已经做过了，得到了推荐的能量截断值。否则，一般来讲一个给定的赝势（模守恒或者超软赝势)，你需要从低到高逐步增加能量截断值来判断总能量已经收敛。

## (e)．布里渊区K点收敛性和什么关系？

答：布里渊区k点只和实空间的晶胞大小有关系，因此当换了晶胞之后需要测试k点的收敛性。然而，一般我们会有一些经验规律，比如实空间某个方向增大一倍，则倒空间相应的k点数量可以少取一半，一直小到只取Gamma点（0，0，0）为止。所以，改变赝势或者交换关联泛函的时候，并不需要重新测试k点的收敛情况。

## (f)．关于长度单位

答：没有注意到单位的转换是初学者容易犯的错之一。长度单位Angstroms（埃）和Bohr的转换关系需要了解（1埃=1.8897259886 Bohr，反过来1Borh=0.529177249 Angstroms，转换时候尽可能多地取有效数字，会让原子坐标更准确)，不同软件的长度单位可能有取得不同的情况，涉及到晶格矢量、原子位置，以及坐标系转换(Direct和Cartesian的区别，前者是晶格坐标系，后者单位才是长度单位)，如果原子位置或者晶胞设错了，不同软件计算结果就会不同。一个小技巧判断不同软件的晶格结构和原子坐标是否一样：可以看不同软件算出来的Ewald 能量（周期性边界条件下算离子-离子库伦排斥能的快速算法)，如果Ewald 能量能对上（精确到小数点后10位以上)，那么说明两个软件的原子坐标和晶格矢量设置都是一样的。

## (g)．关于能量单位

答：涉及到能量单位的有多处。首先是平面波的能量截断值，VASP经常用的是eV，QE 和ABACUS用的是Ry，这两者有转换关系（1Ry=13.605703976eV)，所以用VASP习惯的用户需要把对应的eV转换成Ry再使用ABACUS。其次是 smearing的参数选择，例如0.015，其单位是Ry或者eV也对结果有较大影响。

## 3.基矢量

## (a)．ABACUS支持的基矢量有几种？

答：分成PW（平面波）和LCAO（具体来讲是Numerical atomic orbitals，NAO，即数值原子轨道）两种，目前 ABACUS都支持，像是VASP 和QE 只支持PW，SIESTA 和OpenMP支持 LCAO。还有一种叫lcao_in_pW，就是把NAO 展开成平面波进行计算，但一般只用于测试目的。

## (b)．怎么选择是采用PW还是用LCAO基矢量来做计算？

答：如果是较小原子数的固体体系，建议可以采用PW效率较高。如果是体系原子数较大（例如大于50个原子)，建议可以用LCAO试试，效率会更高。此外，当使用LCAO做计算时，可以先用PW这组相对完备的基组来测试赝势带来的能量收敛性。

## (c)．如果感觉LCAO的效率慢，可能的原因是否和基矢量有关系？

答：ABACUS 里采用的LCAO的局域轨道长度一般在6-10au之间，轨道的截断越长，往往计算精度越高，但计算量和内存消耗也会显著的更大。如果要提高速度，可以在尽可能不影响精度（需要自己测试）的前提下，减小数值原子轨道的半径，这样可以大幅度提升计算效率以及降低内存占用量。

## (d)．ABACUS中LCAO 的 Ecut 是做什么用的

答：LCAO和PW一样，都会有一个 Ecut，在使用LCAO 的时候，增加 Ecut 只会增加局域势格点积分部分的精度（赝势的局域部分+交换关联势+Hartree 势)，并不增加数值原子轨道基矢量的完备性，所以原先用于PW 的收敛性测试方法是不能这样直接用于LCAO的。如果要增加LCAO的基矢量使得基矢量更完备，需要从例如DZP增加到TZDP，而不是增加Ecut。

## (e).LCAO算小分子和固体有什么要注意的？

答：使用局域轨道的话，会有BSSE（Basis set superposition error），对于小分子体系，BSSE 带来的误差会较为严重，使得对于一些分子体系，有必要采用更完备的基矢量（或者更长的局域轨道）或者BSSE 修正。而对于固体系统，一般来讲每个原子会有若干近邻原子，这些近邻原子上的局域轨道会使得在描述每个原子的电子结构时基矢量总体更完备，所以可以相对选择更短的局域轨道。

## (f)．如果算出来SZ结果比DZP好，是否可以用SZ 结果？

答：采用LCAO时，SZ（single zeta 轨道）的结果一般来讲都不收敛，一般不可以用来发表文章。一般至少是DZP 及以上才认为基矢量较为准确，是学界可以认可的NAO 精度。

## 4.赝势

## (a).PAW是不是赝势？

答：准确来说不是，是一种描述电子-离子相互作用的方法。

## (b)．为什么不同软件算出来的同一体系的总能量差距可能会巨大？

答：因为同样的元素，其不同的赝势（或PAW势）允许总能量有一个常数的偏移，因为都对库伦势进行了人为的修改。所以，我们一般在分析结果的时候也不用总能量这个数值，而是用总能量之间的差别，例如bcc相比于fcc 更稳定这种结论，不同赝势哪怕总能量不对齐，其算出来的bcc 和 fcc 之间的能量差也应该是对齐的。另外一件事情就是画态密度（Densityof States，DOS）的时候，我们习惯把费米面取成0点，也是因为总能量是可以有一个常数的偏移的。

(c)．赝势为什么和交换关联泛函有关？答：因为赝势生成过程中解的是一维的Kohn-Sham方程，这个过程也用到了泛函，所以不同泛函生成的赝势就不同。

(d)．赝势改变了，什么参数需要重新测试？答：因为电子和离子的描述方式变了，所以需要重新测试能量截断值 Ecut的收敛性，即增加Ecut，看总能量变化，例如从90Ry变到100Ry，如果总能量变化小于1meV/atom，则认为可以收敛。

(e)．从PBE泛函换到Meta-GGA泛函（如 SCAN）或杂化泛函，是否需要换赝势文件？答：一个约定俗成的赝势用法是：PBE的赝势可以用于Meta-GGA泛函，例如SCAN泛函，甚至杂化泛函包括PBEO 和HSEO6 泛函。但是，LDA 的赝势就不可以用于 SCAN 泛函或者PBEO、HSE06 泛函。因为SCAN泛函和HSE06 泛函也是基于PBE 泛函发展起来的。

(f)．别人文献看到的某元素的能量截断值，是否可以直接使用？答：一般来说是不可以，例如用了VASP的PAW势，就难以直接拿到模守恒赝势里使用。

## 5．电子自洽迭代过程

## (a)．为什么ABACUS的SCF计算比VASP慢很多？

答：首先，先检查Ecut是否一样（注意ABACS的单位是Ry，VASP是eV，要换算)，一般来讲VASP用 PAW方法所需的Ecut比ABACUS 的模守恒赝势要低，目前来讲这部分参数VASP占优，你并不能降低 ABACUS的Ecut来和VASP的Ecut持平。第二，检查你所计算的元素的电子数，往往电子数不同，计算效率会有很大差别。例如Ga的模守恒赝势可能有13个电子，但VASP的Ga的 PAW 势可能只有3个电子，那么VASP的计算效率会快很多也是可以预期的。

## (b)．电子自洽迭代里的展宽技术是用来做什么的？

答：电子自洽迭代过程中使用的展宽（一般称为Smearing）是一种数值技术，它用于提高计算中的收敛稳定性和速度，尤其是在处理金属和其他导体体系时非常有用。展宽通常采用几种不同的办法来实施，例如费米-狄拉克展宽（Fermi-Dirac smearing）、高斯展宽（Gaussian smearing）、Methfessel-Paxton展宽等，它们代表不同的电子占据数分布方法。

## (c)．电子自洽迭代收敛结果和 smearing相关吗？

答：对于没有带隙的金属体系是相关的，采用smearing会在体系总能量里额外引入一个电子熵项。一般来讲 smearing_sigma取得越大越容易收敛，但是要注意的是，smearing_sigma取大了，总能量结果也愈发不准确。因此在迭代接近收敛时，通常需要逐步减少展宽值，以得到更加准确的接近零温的结果。对于半导体来讲，一般较小的 smearing参数不影响总能量结果。

## (d)．只要体系不收敛，就可以调整smearing吗?

答：在项目里对一系列相关体系进行密度泛函理论计算，如果需要用到这些体系的总能量相对值，但发现有体系不收敛，我可以针对不同体系调整不同的 smearing 参数吗？一般不可以，因为 smearing 可能会改变总能量的数值。

## (e)．电荷密度混合是用来做什么的？

答：在电子自洽迭代过程中，需要不断更新电荷密度，直到输入的电荷密度和由此计算得到的输出电荷密度之间的差异足够小，即达到自洽。混合的具体方法有很多种，比如简单的线性混合、Broyden混合、Pulay 混合（也称为 DIS，Direct Inversion in the Iterative Subspace）等。这些方法通过考虑前几次迭代的电荷密度和/或电荷密度差异，来改进当前迭代的电荷密度估计。

## (f)．总能量收敛、受力和应力是否收敛了？

答：一般来讲，总能量在SCF中收敛到几乎不变时，受力和应力不一定完全收敛。如果是对受力或者压力敏感的计算，建议也测试一下相应受力和应力是否收敛（即继续降低scf_thr数值，设置更严格收敛阈值)。

(g).怎么得到准确的带隙？答：对称性、scf、nscf、电荷密度

## 6．原子位置和晶胞结构优化 (弛豫)

## (a)．原子位置和晶胞结构优化有什么需要注意的？

答：首先要注意的就是，原子位置和晶胞结构优化较慢，很耗机时，动辄就是几十上百个进程跑一两天。所以，如果不是很有把握了解你要算的体系，或者不是特别熟练密度泛函理论计算，建议动手算之前找导师聊一聊这样做的可行性，这样做一方面也是可以少走科研上的弯路，另外一方面也能有效的帮助导师节省机时费。

## (b)．原子位置和晶胞结构优化有什么技巧？

答：首先，就是要小心一些参数不要设多了，例如不要包含太多费米面上方非占据的轨道，如果体系没有磁矩，没有必要打开自旋极化计算(nspin=2)。其次，如果你能预估偏离最终的构型较大，那你可以先采用低精度的方法达到一个接近最终构型的一个临时构型。例如，没有必要用18×18×18的k点（虽然你认为这样才能完全收敛总能量）来做优化，你可以先选一个9×9×9的k点mesh来做。第三，如果你是做的一系列相关结构的原子位置和晶胞结构优化，可以想办法构造一些更好的初猜结构，除非考虑亚稳态的存在，否则一开始猜测的结构越接近，所需要的优化步数就越少。第四，BFGS 算法一般适用于偏离稳定结构较小的情况，而cg一般用于偏离稳定结构较大的情况。

## (c)．做结构优化很多步都不收敛，怎么办？

答：首先，检查每一步的SCF 是否收敛，如果不收敛，先让SCF能收敛，再做结构优化。其次，如果能不用cel-relax而只用relax，建议尽量先用 relax，把晶胞内的原子位置先固定住了，最后再做cell-realx，这样效率更高。原因是如果把cell的自由度和原子的自由度一起考虑做relax，往往效率较低。

## (d)．采用LCAO和PW，结构优化收敛有什么区别？

答：PW基组更完备，一般结构优化可以收敛到0.001eV/Angstroms。而LCAO 由于基矢量不够完备,存在诸如egg box effect 的误差，所以求力能收敛到的精度一般要差一些，通常 0.04 eV/Angstroms 以下是可以接受的，也有不少文章采用这个精度来发表结果，是被认可的。

## 7.并行

## （a)．MPI并行和OpenMP并行的区别是什么？什么叫进程，什么叫线程？

答：MPI进程和OpenMP线程的差别，前者内存是不共享的，后者内存是共享的。

## (b)．原子位置和晶胞结构优化有什么技巧？

答：首先，就是要小心一些参数不要设多了，例如不要包含太多费米面上方非占据的轨道，如果体系没有磁矩，没有必要打开自旋极化计算(nspin=2)。其次，如果你能预估偏离最终的构型较大，那你可以先采用低精度的方法达到一个接近最终构型的一个临时构型。例如，没有必要用18×18×18的k点（虽然你认为这样才能完全收敛总能量）来做优化，你可以先选一个9×9×9的k点mesh来做。第三，如果你是做的一系列相关结构的原子位置和晶胞结构优化，可以想办法构造一些更好的初猜结构，除非考虑亚稳态的存在，否则一开始猜测的结构越接近，所需要的优化步数就越少。第四，BFGS 算法一般适用于偏离稳定结构较小的情况，而cg一般用于偏离稳定结构较大的情况。

## (c)．是不是进程或者线程越多，计算就越快？

答：举个例子，用7au的DZP轨道算512个Fe原子的体系，k点1×1×1，以下是运行时间（OMP代表线程数，MPI代表进程数)：1OMP×32 MPI=262s 每电子步；1OMP×64 MPI=581s 每电子步；1OMP×128 MPI=654s每电子步；1OMP×256MPI=765s 每电子步。可以看到进程越多，计算反而越慢，这是因为进程间的通讯量随着进程数增多变大了。所以，并不是用的计算资源越多，计算就会越快，建议做计算前先评估好并行效率。

## (d)．k点并行（也就是KPAR这个参数)，要怎么理解它的使用？

答：对PW基组计算来说，不同k 点的Kohn-Sham方程求解（即寻找特征值和特征向量）是独立的,在给定总进程数的前提下，ABACUS提供把不同k 点的方程求解并行的方案，通过参数KPAR 实现，该参数代表要将k 点分成几组，原则上KPAR 越大效率越高，但KPAR不应超过进程数或者k点数。如果 KPAR设成1，则程序执行时只对PW进行并行。在有些情况下，仅用PW并行并行效率也挺高，因此看不出k点并行的优势。

## 第2章编译ABACUS

## 2.1GCC 编译 ABACUS 教程

## 2.1.1介绍：关于ABACUS

ABACUS（原子算筹）软件同时支持两种基矢量，一种是平面波（Plane wave，PW）基组，一种是LCAO(Linear combination of atomic orbitals)或者称 NAO(Numerical atomic orbitals)基组。因为两种基组运行依赖的软件库不完全相同，所以我们分开介绍。

实际上，LCAO基组依赖的软件库比PW基组更多，因此如果用户只使用PW基组，并不需要将所有依赖的数学库全都下载安装好。

下面介绍，两种基组分别都依赖那些数学库：

。 PW基组依赖以下3个数学库：

BLAS：BLAS（Basic Linear Algebra Subprograms）是一组基本的线性代数库，用于高效执行常见的基础线性代数运算，如矩阵乘法和向量操作。

LAPACK：LAPACK（Linear AlgebraPackage）是一个开源的数值线性代数库，用于解决线性代数问题，包括矩阵分解、方程组求解和特征值计算等。

FFTW3:FFTW3（Fastest Fourier Transform in the West3）是一个高性能、开源的快速傅里叶变换（FFT)库，用于高效计算各种傅里叶变换和逆变换。

。 LCAO基组依赖以下6个软件库：

BLAS

LAPACK

FFTW3

ScaLAPACK：ScaLAPACK（Scalable LAPACK）是一种并行计算库，构建在LAPACK之上，用于解决大规模线性代数问题，特别适用于分布式和并行计算环境。

CEREAL：CEREAL是一个C++ 序列化库，用于将C++ 对象转换为可存储或传输的数据格式，以及将其反序列化回对象。

ELPA：ELPA（Eigenvalue SoLvers for Petaflop-Applications）是一个用于高性能计算的开源库，旨在解决大规模高性能计算中的特征值问题，特别是密集矩阵的特征值问题。（用户编译安装abacus 的时候大多在这里出现问题)

ELPA 仅用于LCAO 基组求解特征值与特征矩阵的过程，如果由于机器等原因导致ELPA难以安装，可以暂不安装ELPA，在编译abacus 的时候添加-DUSE_ELPA=O 即可绕过ELPA安装困难的问题。

在 ABACUS 中，LCAO基组求解特征值与特征矩阵提供了两种方法，一种是调用ELPA 进行求解（默认选项），一种是调用 ScaLAPACK进行求解。

如果使用-DUSE_ELPA=0编译选项，请参考使用文档配置ks_solver为 scalapack_gvx，调用 ScaLAPACK 进行求解。

## 2.1.2检查当前服务器基础环境：

## 1.查看当前系统版本

1 root@bohrium-11852-1041346:\~# lsb_release -a   
2 No LSB modules are available.   
3 Distributor ID: Ubuntu   
4 Description:Ubuntu 20.04.5 LTS

```yaml
5 Release: 20.04
6 Codename: focal
```

## 2.检查当前GCC编译器版本

1 root@bohrium-11852-1041346:\~# g++ --version   
2 g++ (Ubuntu 9.4.0-1ubuntu1\~20.04.1) 9.4.0   
3 Copyright (C) 2019 Free Software Foundation， Inc.

## 3.检查当前环境是否有git

1 root@bohrium-11852-1041346:\~# git --version   
2 git version 2.25.1

## 2.1.3安装仅支持PW基组的ABACUS

对于仅支持PW基组的ABACUS，我们同时支持两种版本的编译：串行版本和并行版本。编译这两个版本的主要区别在于是否需要安装 MPIlibrary。

## 1.编译串行版本

## 安装依赖库

```shell
1 sudo apt update
2 sudo apt install -y libopenblas-openmp-dev
3 sudo apt install -y liblapack-dev
4 sudo apt install -y libfftw3-dev
```

## 从仓库克隆代码到本地，并进人目录

1 git clone https://github.com/deepmodeling/abacus-develop.git   
2 cd abacus-develop/

## 开始编译并安装abacus

```shell
1 cmake -B build -DENABLE_MPI=OFF
2 cd build && make -j'nproc'
```

可以看到在当前目录下，已经编译成功了abacus_pw_serial可执行文件,abacus_pw_serial就是串行版本的支持平面波基组的ABACUS。

## 2.编译并行版本

安装依赖库 (同串行，如果已经安装，则不用重复操作)

```shell
1 sudo apt update
2 sudo apt install -y libopenblas-openmp-dev
3 sudo apt install -y liblapack-dev
4 sudo apt install -y libfftw3-dev
```

安装MPIlibrary,这里选择更为常用的open MPI

1 sudo apt install -y libopenmpi-dev

从仓库克隆代码到本地，并进人目录

1 git clone https://github.com/deepmodeling/abacus-develop.git   
2 2 cd abacus-develop/

开始编译并安装abacus

```batch
1 cmake -B build -DENABLE_LCAO=OFF
2 cd build && make -j'nproc"
```

可以看到在当前目录下，已经编译成功了abacus_pw可执行文件,abacus_pw就是并行版本的支持平面波基组的 ABACUS。

## 2.1.4安装支持两种基组的ABACUS

对于同时支持两种基组的ABACUS，仅可以编译并行版本。并无串行版本。

在本文最初就提到支持LCAO 基组的ABACUS 所依赖软件库的更多。

1.安装PW基组依赖的软件库。(同上，如果已经安装，则不用重复操作)

```shell
1 sudo apt update
2 sudo apt install -y libopenblas-openmp-dev
3 sudo apt install -y liblapack-dev
4 sudo apt install -y libfftw3-dev
```

2.安装MPIlibrary。这里选择更为常用的open MPI。(同上，如果已经安装，则不用重复操作)

1 sudo apt install -y libopenmpi-dev

3.安装LCAO 基组依赖的 ScaLAPACK和CEREAL软件库。

1 sudo apt install -y libscalapack-mpi-dev   
2 sudo apt install -y libcereal-dev

## 4.安装ELPA软件库(一般这里容易出问题)

如果你的系统是Ubuntu22.04，那么你可以以很简单的方式成功安装ELPA：

1 sudo apt install -y libelpa-dev

如果你的系统不是Ubuntu 22.04，很可惜，你需要稍微辛苦一点来安装 ELPA。但别急，这也并不难！手动编译安装ELPA可以分为以下几步：

·下载并进入ELPA目录：

1 wget https://elpa.mpcdf .mpg.de/software/tarball-  
2 archive/Releases/2021.05.002/elpa-2021.05.002.tar.gz   
3 tar xzf elpa-2021.05.002.tar.gz   
4 cd elpa-2021.05.002   
5 mkdir build && cd build

## • configure

```shell
1 ../configure --enable-openmp CFLAGS="-03 -march=native -funsafe-
2 lop-optimizations -funsafe-math-optimizations -ftree-vect-loop-
3 version -ftree-vectorize" FCFLAGS="-02 -mavx" --disable-avx512
```

## ·编译安装

1 make -jnproc^   
2 make install   
3 ln -s /usr/local/include/elpa_openmp-2021.05.002/elpa /usr/local/include/

注意：ln -s /usr/local/include/elpa_openmp-2021.05.002/elpa/usr/local/include/是非常重要的！（很多用户是这里出的问题！)

## 5.开始编译并安装abacus

```batch
1 cmake -B build
2 cd build && make -j'nproc"
```

可以看到在当前目录下，已经编译成功了abacus 可执行文件。

## 2.2 Intel oneAPI 2024/2025 编译 ABACUS 教程

## 2.2.1 Intel OneAPI工具链

## 1.简介

本教程介绍如何使用InteloneAPI2024/2025工具链编译ABACUS。

## OneAPI Base Toolkit

OneAPI Base Toolkit提供了一系列基础工具和库，包括 BLAS,LAPACK,ScaLAPACK and FFTW3 等关键组件：

，Intel? oneAPI DPC++/C++ Compiler：面向 CPU、GPU、FPGA 的 C++ 编译器。

Intel? oneAPIDPC++Library：提供并行算法的库。

·Intel@ oneAPIMath Kernel Library：即MKL，提供高度优化的数学函数库，包括线性代数、FFT等。

。 Intel?VTuneTMProfiler：性能分析优化工具。

## OneAPI HPC Toolkit

OneAPI HPC Toolkit 是为高性能计算（HPC）特别优化的工具集，它在OneAPIBase Toolkit 的基础上增加了更多针对HPC应用的特性和工具，包括：

。 Intel? oneAPIDPC++/C++Compiler:含有C++ MPI编译器。

。Intel? Fortran Compiler & Intel? Fortran Compiler Classic:Fortran 编译器。

Intel@ Intel@ MPI Library:MPI库。

。 Intel?VTuneTMProfiler：性能分析优化工具。

## 2.安装

可以从官网上获得最新的安装包。安装需要管理员权限。

Download the Intel? oneAPI Base Toolkit

Download the Intel? HPC Toolkit

以2024.2版本为例，展示配置环境和安装ABACUS 的流程。

## Offline installer

## Base Toolkit

。 下载

1 wget https://registrationcenter-download.intel.com/akdlm/IRC\_NAS/9a98af19-1c68-46ce-9fd-e24   
2 9240c7c42/1\_BaseKit\_p\_2024.2.0.634\_offline.sh

## 。 安装，可选图形化界面安装

```batch
1 sudo sh ./1\_BaseKit\_p\_2024.2.0.634\_offline.sh
```

## 。 或直接在 shell中静默安装

```batch
1 sudo sh ./l\_BaseKit\_p\_2024.2.0.634\_offline.sh -a --silent --cli --eula accept
```

## Base Toolkit

。 下载

1 wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d4e49548-1492-45c9-b678-8268   
2 cb0f1b05/l_HPCKit_p_2024.2.0.635_offline.sh

。 安装，类似

```batch
1 sudo sh ./l_HPCKit_p_2024.2.0.635_offline.sh -a --silent --cli --eula accept
```

·按照默认配置，全局安装目录位于/opt/intel/oneapi。以2024.2版本为例，该目录如下：

oneapi ls   
2 2024.2 basekit common dal dev-utilities dnnl dpl installer ippcp logs   
3 modulefiles-setup.sh setvars.sh tbb vtune advisor ccl compiler debugger diagnostics   
4 dpcpp-ct hpckit ipp licensing mkl mpi support.txt tcm

。 其中有一些重要的目录和工具，如2024.2/和 installer/

## 3.环境变量设置

要使用oneAPI提供的编译器和库，需要正确设置环境。

2024.0及之后的版本，使用了新的目录布局。这导致了和老版本不同的环境配置方式。参见Use the setvarsand oneapi-vars Scripts with Linux\*

和之前的版本相比，由组件目录布局（Component Directory Layout）改为统一目录布局（Unified DirectoryLayout），新版本的所有组件（bin,lib,include,share）等统一放在以工具包版本号命名的顶级目录中。在原先的Component Directory Layout 中，不同的组件有各自的环境变量设置脚本，由一个位于oneAPI安装目录的脚本/opt/intel/oneapi/setvars.sh统一管理。新版的 Unifed Directory Layout 中，每个组件被集中到组件共用的共享文件夹中，即每个组件将其头文件提供给一个公共的 include 文件夹，将其库文件提供给一个公共的 lib 文件夹，以此类推。这样，不同版本工具包之间的切换更容易，无需维护通用的 setvars.sh，而是通过工具包版本号命名的目录提供的脚本 /opt/intel/oneapi/<toolkit-version>/oneapi-vars.sh 设置。

以 2024.2为例，每次使用 icpx 等编译器之前，需要在 shell环境中 source一次脚本:

1 . /opt/intel/oneapi/2024.2/oneapi-vars.sh   
2 # configure, build,...

此时，编译器应该被添加到环境中，可以运行命令检查是否正确配置：

1 # source 前   
2 -> oneapi mpiicpx -v   
3 zsh: command not found: mpiicpx   
4 # source 后   
5 -> oneapi mpiicpx -v   
mpiicpx for the Intel(R) MPI Library @IMPI_OFFICIALVERSION@ for Linux\*   
Copyright Intel Corporation.   
8 Intel(R） oneAPI DPC++/C++ Compiler 2024.2.0 (2024.2.0.20240602)   
9 Target: x86_64-unknown-linux-gnu   
10 Thread model: posix   
11 InstalledDir: /opt/intel/oneapi/compiler/2024.2/bin/compiler   
12 Configuration file: /opt/intel/oneapi/compiler/2024.2/bin/compiler/../icpx.cfg   
13 Found candidate GCC installation: /usr/lib/gcc/x86_64-linux-gnu/11   
14 Selected GCC installation: /usr/lib/gcc/x86_64-linux-gnu/11   
15 Candidate multilib: .;@m64   
16 Selected multilib: .;@m64

```yaml
17 icpx: warning: argument unused during compilation: '-I /opt/intel/oneapi/2024.2/include'
18 [-Wunused-command-line-argument]
```

·．命令在POSIX 标准中定义，因此它在所有POSIX 兼容的 shell 中都应该可用。

· source 命令一般情况下是.的别名，但在某些环境中不可用。如果脚本不能使用 source，请换成通用的..

这一步会配置环境变量，但是并不会设定默认的C++编译器。因此，设置了oneAPI编译器的环境变量之后，还需要在配置（configure）时指定构建（build）采用的编译器，如CXX=icpx。

2024.0开始，原先的classical编译器停止使用。

请使用新的 icpx/mpiicpx取代原来的 icpc/mpiicpc。

## 2.2.2 ABACUS安装

## 1.依赖库

安装cereal

1 sudo apt install libcereal-dev

## 安装 elpa

在 Ubuntu22.04 等发行版中，可以通过apt获取预构建的elpa包（pre-build packages)。不幸的是，使用不同的MPI库构建elpa可能会导致冲突，apt获取的elpa包和oneAPI2024.2构建的abacus 在运行时可能出现问题。

1 /usr/bin/ld: warning: libmpi.so.40, needed by /usr/lib/x86_64-linux-gnu/libelpa.so, may conflict   
2 with libmpi.so.12

## 运行算例时报错：

1\*\*\*\*\*\*   
2 << Start SCF iteration.   
3 Abort(403251971) on node O (rank O in comm O): Fatal error in internal_Bcast: Unknown error class,   
4 error stack:   
5 internal_Bcast(4152): MPI_Bcast(buffer=0x2c0e960， count=1， INVALID DATATYPE， 1， comm=0xc40001b)   
6 failed   
internal_Bcast(4112): Invalid datatype   
8 Abort(67707651) on node 1 (rank 1 in comm O): Fatal eror in internal_Bcast: Unknown error class,   
9 error stack:   
10 internal_Bcast(4152): MPI_Bcast(buffer=0x2fcdac0， count=1， INVALID DATATYPE， 1， comm=0xc400013)   
11 failed   
12 internal_Bcast(4112): Invalid datatype

因此，我们需要手动利用oneAPI工具链构建elpa。

## 使用 toolchain

由于elpa 的编译较复杂，我们可以通过 abacus toolchain 脚本自动构建和安装 elpa。该脚本用icpx 和 mpiicpc编译elpa，同时会默认安装cereal和libxc。

一键配置编译 ABACUS|toolchain 脚本的使用

---

## 第 21–40 页

1 . /opt/intel/oneapi/2024.2/oneapi-vars.sh  
2 cd abacus-develop/toolchain  
3 ./toolchain_intel.sh # 脚本将会用 intel 工具链安装依赖。

安装完成的库目录为abacus-develop/toolchain/install.

1 # 安装完成的库目录为 abacus-develop/toolchain/install   
2 -> toolchain git:(develop) ls install   
3 cereal-1.3.2 cmake-3.28.1 elpa-2023.05.001 libxc-6.2.2 lsan.supp setup toolchain.conf   
4 toolchain.env tsan.supp valgrind.supp

使用toolchain 构建的elpa，在构建 abacus 时有警告如下，暂未发现影响使用。该问题由 scalapack库未指定mkl 版本导致，“自行编译安装”方式（见下）无此问题。

参见2.4 Non standard paths or non standard libraries

1 /usr/bin/ld: warning: libmpi.so.40, needed by /lib/x86_64-linux-gnu/libscalapack-openmpi.so.2.1,   
2 may conflict with libmpi.so.12

除了使用 toolchain，也可以根据 documentation/INSTALL.md·master·elpa/elpa·GitLab，自行编译安装elpa。

## 自行编译安装

自行下载和安装elpa 到环境中。以oneAPI2024.2 和elpa-2024.05.001为例。

·在官网下载elpa 包。

1 vget https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2024.05.001/elpa-2024.05.001.tar.gz   
3 tar xvf elpa-2024.05.001.tar.gz

·以oneAPI2024.2为例提供一个安装脚本，仅供参考。这一步可能需要几分钟。如果运行脚本时显示oneAPIwarnings提示已经设置过环境变量并退出，请打开一个新的 shell环境，执行安装脚本。最后的make install和 ln 需要 sudo权限，如果无法提供，请手动完成这两步。

。 oneAPI2025.0 开始，Intel? Fortran Compiler Classic (ifort)停止使用。

请使用新的 ifx/mpiifx取代原来的 ifort/mpiifort。

请根据实际情况，指定安装脚本中的oneAPI工具链版本。如果使用oneAPI2025.0之后的版本，请把 21行FC=mpiifort 替换为 FC=mpiifx。

```batch
1 cd elpa-2024.05.001
```

```shell
1 #!/bin/bash -e
2
3 # buildelpa.sh
4 # run in elpa main dir
5
6 # source oneAPI environments
7 echo "using oneAPI 2024.2"
```

```shell
8 . /opt/intel/oneapi/2024.2/oneapi-vars.sh \
9 Il { echo "Failed to load oneAPI environment. Please restart in a new shell without oneAPI vars
10 set."; false; }
11
12 # in elpa main dir
13 # check vhether there is a 'build' directory
14 if [ -d "build" ]； then
15 echo "rm -rf build"
16 rm -rf build
17 fi
18 mkdir build \&\& cd build
19
20 MKL_H0ME=/opt/intel/oneapi/2024.2
21
22 __CC=mpiicx CXX=mpiicpx FC=mpiifort ../configure \
23 --disable-avx --disable-avx2 --disable-avx512 --disable-sse --disable-sse-assembly \
24 SCALAPACK_LDFLAGS="-L\$MKL_H0ME/lib/ -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
25 -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -Wl,-rpath,
26 \$MKL_HOME/lib/" \
27 SCALAPACK_FCFLAGS="-L\$MKL_H0ME/lib/ -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential \
28 -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -I\$MKL_H0ME/include/mkl/
29 intel64/lp64"
30
31 make -j\$(nproc) > make.log 2>\&1
32 echo "installation process may require administrative privileges."
33 read -p "Would you like to continue with 'sudo make install'? (y/n): " -n 1 -r
34 echo  # (Optional） Move to a new line
35
36 if [[ \$REPLY =~ ^[Yy]\$ ]]
37 then
38 echo "Proceeding with installation using administrative privileges..."
39 sudo make install > install.log 2>\&1
40 else
41 echo "Installation has been canceled."
42 echo "Please manually execute 'make install' and link elpa to default path."
43 fi
44
45 # link elpa to /usr/local/include
46
47 # The target path for the symbolic link
48 LINK_PATH="/usr/local/include/elpa"
49 # The source path for the original link (replace with the actual source path)
50 S0URCE_PATH="/usr/local/include/elpa-2024.05.001/elpa"
51
52 # Check if the link exists
53 if [ -L "\$LINK_PATH" ]； then
54 # If the link exists,delete it
55 echo "The link already exists，deleting the old link..."
```

```shell
56 sudo rm "\$LINK_PATH"
57 else
58 # If the link does not exist， check for the presence of a file or directory
59 if [ -e "\$LINK_PATH" ]； then
60 echo "A file or directory exists at the path, unable to create the link.
61 Please delete or rename the file/directory first."
62 exit 1
63 fi
64 fi
65
66 # Create a new symbolic link
67 SUdo ln -s "\$SOURCE_PATH" "\$LINK_PATH"
68
69 # Check if the link was created successfully
70 if [ -L "\$LINK_PATH" ]； then
71 echo "The new symbolic link has been created successfully."
72 else
73 echo "Failed to create the symbolic link."
74 exit 1
75 fi
76
77 echo "elpa install over."
```

## 2.安装ABACUS

配置好依赖后，我们可以开始安装ABACUS。

如果你使用toolchain配置环境，那么toolchain提供了一系列环境变量和构建工具。构建时，请利用abacus-develop/toolchain/build_abacus_intel.sh脚本直接构建ABACUS（在其中修改配置选项)，也可以手动构建。（不推荐)

如果需要手动安装 ABACUS 并使用 toolchain 的 ELPA：需要记住此前 toolchain 安装 elpa的目录。  
如果之前选择手动安装ELPA，可以用默认方式自行用cmake 构建。

```shell
1#设置环境
2 . /opt/intel/oneapi/2024.2/oneapi-vars.sh
3 # configure
4 # 在此选择oneAPI 的编译器，添加编译选项，指定此前的安装路径，如
CXX=mpiicpx cmake -B build \
6 -DELPA_DIR=~/abacus-develop/toolchain/install/elpa-2023.05.001/cpu/
7 # 根据需要添加其他 cmake 选项
8
9 # build
10 cmake --build build -j'nproc`
11 # install
12 cmake --install build
```

## 2.2.3常见问题和解决方法

在开始构建之前，请清除原有的build目录。

1 rm -rf build

找不到编译器，记得运行 vars 脚本设置oneAPI环境变量。

CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineCXXCompiler.cmake:48 (message):   
Could not find compiler set in environment variable CXX:   
3   
4 mpiicpx.   
5   
6 Call Stack (most recent call first):   
CMakeLists.txt:7 (project)   
8   
9   
10 CMake Error: CMAKE_CXX_COMPILER not set， after EnableLanguage   
11 -- Configuring incomplete， errors occurred!   
12   
13 # 请设置环境   
14 . /opt/intel/oneapi/2024.2/oneapi-vars.sh

找不到elpa，请在配置时指定安装路径。

CMake Error in CMakeLists.txt:   
2 Imported target "ELPA::ELPA" includes non-existent path   
3   
4 "/usr/include/elpa"   
5   
6 in its INTERFACE_INCLUDE_DIRECTORIES. Possible reasons include:   
7   
8 \* The path was deleted， renamed， or moved to another location.   
9   
10 \* An install or uninstall procedure did not complete successfully.   
11   
12 \* The installation package was faulty and references files it does not   
13 provide.   
14   
15   
16# 请指定 ELPA_DIR（及其他自己手动构建的库）   
17 CXX=mpiicpx cmake -B build \   
18 -DELPA_DIR=\~/abacus-develop/toolchain/install/elpa-2023.05.001/cpu/

期望使用 intel工具链编译，但cmake 显示使用GNU工具链。请使用CXX=mpiicpx 指定编译器。

```shell
cmake -B build -DELPA_DIR=~/abacus-develop/toolchain/install/elpa-2023.05.001/cpu/
2 -- The CXX compiler identification is GNU 11.4.0
3
4 # CxX=mpiicpx cmake ...
5 # -- The CXX compiler identification is IntelLLVM 2024.2.0
```

cmake 提示编译器错误，请使用新版的 icpx/mpiicpx，而不是 icpc/mpiicpc

CMake Error at /usr/share/cmake-3.22/Modules/CMakeTestCXxCompiler.cmake:62 (message):   
The C++ compiler   
"/opt/intel/oneapi/2024.2/bin/mpiicpc"   
5   
6 is not able to compile a simple test program.   
8 # use Cxx=mpiicpx instead of Cxx=mpiicpc

运行了oneAPI配置环境变量脚本，但是链接错误

请检查oneAPIHPC kits 的安装；可进人 installer查看当前安装的所有 Toolkits 和对应版本，见下条。

如果安装了多版本的oneAPI工具链，怀疑环境遭到破坏，可以使用/opt/intel/oneapi/instaler中的 installer工具修复和移除不需要版本以及更新。

```shell
1 cd /opt/intel/oneapi/installer
2 sudo ./installer
3
4 # 可以用 Repair 尝试修复环境
5 # 使用 Remove 移除不需要的组件
6# 使用 Update 获得新版本
```

如果遇到 libmpi.so 相关报错，可以用locate查看所有相关库。

1 locate libmpi.so

运行算例或测试失败，请确保最新构建后运行了install命令，且没有因为权限不足安装失败。

## 2.2.4 参考

Abacus 文档   
Easy Installation   
一键配置编译 ABACUS|toolchain 脚本的使用   
InteloneAPI编译ABACUS教程·GitBook   
elpa   
documentation/INSTALL.md ·master · elpa/elpa · GitLab   
Intel 文档   
Use the setvars and oneapi-vars Scripts with Linux\*   
Intel? C++ Compiler Classic Release Notes   
Porting Guide for DPCPP or ICX   
Porting Guide for ifort Users to ifx   
Intel? oneAPI Base Toolkit Release Notes

## 2.3Intel oneAPI编译 ABACUS教程

## 2.3.1 前言

非特殊情况，Linux 系统会默认安装GCC（GNUCompiler Collection）编译器套件。因此如果你的机器默认没有Intel?oneAPI Toolkits，那么建议，你可以直接看2.1：GCC 编译ABACUS 教程。

如果您的机器恰好有Intel?oneAPIToolkits，那么您也可以使用它来编译 ABACUS。

## 2.3.2 关于 Intel? oneAPI Toolkits

Intel? oneAPI Tolkits 分为很多个版本的 Toolkit，这里解释下 Intel? oneAPI Base Toolkit 和Intel? oneAPIHPC Toolkit 的区别。

简单来说：Intel? oneAPIHPC Toolkit 包含 Intel? oneAPI Base Toolkit。

Intel® oneAPI Base Toolkit 中包含：Intel? oneAPIMath Kernel Library（简称为 MKL），这个软件库里面包含了大量常用的数学库，包括ABACUS 依赖的BLAS、LAPACK、FFTW3和 ScaLAPACK4个软件库。

因此使用InteloneAPI来编译ABACUS，可以省去逐个去安装数学库的过程！

Intel® oneAPIHPC Toolkit 相比于Intel@ oneAPI Base Toolkit多了一些必要的Intel的编译器以及 Intel@ MPILibrary（ABACUS 想要编译并行版本必须的关键库）。

总结一下，使用 Intel oneAPI编译 ABACUS，必须确保你的机器中包含 Intel@ oneAPIHPC Toolkit。

关于Intel?oneAPI Toolkits 更详细的信息：

https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit

## 2.3.3编译安装

开启Intel? oneAPI Toolkits:

```shell
1 source /opt/intel/oneapi/setvars.sh
```

ELPA 仅用于LCAO 基组求解特征值与特征矩阵的过程，如果由于机器等原因导致 ELPA 难以安装，可以暂不安装 ELPA,在编译 abacus 的时候添加-DUSE_ELPA=0即可绕过ELPA安装困难的问题。在ABACUS 中,LCAO基组求解特征值与特征矩阵提供了两种方法，一种是调用 ELPA 进行求解（默认选项)，一种是调用 ScaLAPACK进行求解。如果使用-DUSE_ELPA=0编译选项,请参考使用文档配置ks_solver为scalapack_gvx,调用ScaLAPACK进行求解。(http://abacus.deepmodeling.com/en/stable/advanced/input_files/input-main.html#ks-solver)

## 1.安装仅支持PW基组的ABACUS

对于仅支持PW 基组的 ABACUS，我们同时支持两种版本的编译：串行版本和并行版本。

ABACUS 的PW基组只依赖:BLAS、LAPACK、FFTW3三个数学库，而这三个数学库都已经被 Intel@ oneAPIBase Toolkit 中包含：Intel? oneAPI Math Kernel Library（简称为MKL）所包含，因此我们不需要再安装其他的软件库！

## 编译串行版本

1 CXX=icpx cmake -B build -DENABLE_MPI=OFF   
2 cd build && make -j'nproc"

注意，这里必须指定CXX=icpx，不然系统会使用默认的CXX 编译器。同时这里的CXX也不要指定 icpc,icpc 是 Intel? C++Compiler Classic，会带来一些报错。

可以看到在当前目录下，已经编译成功了abacus_pW_serial可执行文件。

abacus_pW_serial就是串行版本的支持平面波基组的ABACUS。

## 编译并行版本

```shell
1 CXX=icpx cmake -B build -DENABLE_LCAO=OFF
2 cd build && make -j'nproc'
```

可以看到在当前目录下，已经编译成功了abacus_pw 可执行文件。

abacus_pw 就是并行版本的支持平面波基组的ABACUS。

## 2.安装支持两种基组的ABACUS

对于同时支持两种基组的ABACUS，仅可以编译并行版本。并无串行版本。

完整版的 ABACUS 需要依赖更多的软件库：BLAS、LAPACK、FFTW3、ScaLAPACK、CEREAL、ELPA共6个。

而前四个数学库（BLAS、LAPACK、FFTW3、ScaLAPACK）都已经被 Intel MKL所包含，因此我们只需再额外安装 CEREAL、ELPA 即可。

安装CEREAL

1 sudo apt install -y libcereal-dev

## 安装ELPA

如果你的系统是Ubuntu22.04，那么你可以以很简单的方式成功安装ELPA。

1 sudo apt install -y libelpa-dev

如果你的系统不是Ubuntu22.04，很可惜，你需要稍微辛苦一点来安装 ELPA。但别急，这也并不难!注意；这里手动编译安装 ELPA的方法与在GCC下编译不太一样，因为要保证ELPA 和ABACUS都是基于Intel oneAPI来编译的！

手动编译安装ELPA可以分为以下几步：

下载并进入ELPA目录：

wget htps://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2021.05.002/elpa-2021.05.002.tar.gz   
2 tar xzf elpa-2021.05.002.tar.gz   
3 cd elpa-2021.05.002   
4 mkdir build && cd build

configure

1 CC=mpiicc CXX=mpiicpc FC=mpiifort ../configure --enable-openmp FCFLAGS="-qmkl=cluster"

## 编译安装

```shell
1 make -j`nproc'
2 make install
3 ln -s /usr/local/include/elpa_openmp-2021.05.002/elpa /usr/local/include/
```

注意：In-s /usr/local/include/elpa_openmp-2021.05.002/elpa/usr/local/include/是非常重要的！（很多用户是这里出的问题!)

编译安装ABACUS

```shell
1 CXX=icpx cmake -B build
2 cd build && make -j'nproc
```

可以看到在当前目录下，已经编译成功了abacus 可执行文件。

abacus 就是完整版的ABACUS，它同时支持两种基矢量，而且还是并行版本！

## 2.4 Toolchain 编译 ABACUS 教程(1) -GCC

## 2.4.1 简介

从前面的教程中我们可以发现，ABACUS 的源码编译涉及到众多依赖库，如 CMake 编译管理器，C/C++/Fortran 编译器，MPI并行库、数学库、泛函库和其他功能依赖库（如LCAO基组，杂化泛函计算，abacus-json 支持）等，从源码编译需要正确编译这些依赖库，并在编译ABACUS 时正确声明各个依赖库，对用户的软件编译能力提出了挑战。尤其对于广大对 Linux并不十分熟悉的用户而言，编译软件会变得比使用软件更加困难。

那么，有相对简单高效的办法吗，答案是有的，这就是ABACUSToolchain。

## 2.4.2什么是 ABACUS Toolchain

ABACUS Toolchain 是一套内置于 ABACUS 仓库中的 bash 脚本集，它能帮助用户在线或离线地编译安装ABACUS 所需的软件依赖，自动处理各个依赖库的环境变量，并基于这些依赖库快速完成 ABACUS 源码编译过程。本教程基于 2025-02 版本的 ABACUS Toolchain 撰写，目前，ABACUS Toolchain 支持如下编译安装功能：

·GCC Toolchain，即从足够版本的GCC（也称GNU）编译套件（gcc,g++,gfortran）出发，从头编译安装ABACUS 依赖库及ABACUS 本体的 Toolchain方法。

·Intel Toolchain，即基于Intel的编译器、数学库和并行库（通常打包在Intel-OneAPI或 Intel-paralel-xe-studio内）编译安装 ABACUS 依赖库及ABACUS 本体的 Toolchain方法。

·AMD Toolchain，即基于AMD 的编译器和数学库进行 ABACUS 编译安装的方法，细分为两种：

·GCC-AOCL Toolchain，即基于系统 gcc 编译器和 AMD 的 AOCL 数学库编译安装 ABACUS 依赖及其本体。

·AOCC-AOCL Toolchain，即基于 AMD 的 AOCC 编译器和 AOCL 数学库编译安装 ABACUS 依赖及其本体。

同时，ABACUS Toolchain 还支持一些高级功能，如：

·支持几乎所有 ABACUS 的额外功能插件的编译（如 LibRI,DeePKS,DeePMD 等）。

。 离线安装 ABACUS 依赖库和 ABACUS 本体。

安装GPU支持的 ABACUS。

·使用系统已有的依赖库配合Toolchain下载的依赖库安装 ABACUS（可通过环境变量或绝对路径指定两种方法)

·将ABACUS 的依赖库安装到指定位置。

。 切换不同的ABACUS 依赖库版本.

总体来说，ABACUS Toolchain 希望达到的愿景是：

·方便用户高效地从源码编译最适合当前服务器环境的ABACUS,并能快速测试不同依赖库类型的Toolchain编译所得ABACUS 的计算效率。

·建立一套 ABACUS 源码编译的标准流程，ABACUS开发者可以直接在 Toolchain 中控制 ABACUS 各个依赖库的版本和编译方式，而不需要自行编译并手动加入各种编译选项。与此同时，如果 ABACUS 的安装依赖有所更新，开发者也可以在Toolchain中直接更新相关信息，即可为用户使用。

## 2.4.3基于 GCC Toolchain 安装 ABACUS

下面我们将以GCC Toolchain为例，展示安装LTS-3.10.0版本 ABACUS 的过程，并让各位用户快速上手Toolchain 使用。GCC Toolchain 也是基于 Toolchain 的 ABACUS 编译安装方法中通用性最好的。

1.下载ABACUS仓库：可以通过gitclone下载，这一做法允许用户通过git包管理器快速更新或切换版本。也可以通过 wget拉取ABACUS仓库压缩包，这样得到的ABACUS 仓库没有git相关的信息，文件体积更小。

1 # via github   
2 git clone https://github.com/deepmodeling/abacus-develop.git -b LTS   
3 # via wget from github by codeload: can be used under CN Internet   
4 wget https://codeload.github.com/deepmodeling/abacus-develop/tar.gz/LTS -0 abacus-LTS.tar.gz   
5 tar -zxvf abacus-LTS.tar.gz

## 2.进入Toolchain目录

1 cd abacus-develop/toolchain   
2 # if you download abacus via wget from codeload   
3 cd abacus-develop-LTS/toolchain

## 3．运行 toolchain_gnu.sh 脚本

1 sh ./toolchain_gnu.sh

如此即可开始基于Toolchain的ABACUS 依赖库编译安装。注：在使用 Toolchain 安装之前，务必确保你的服务器相关环境变量是基本干净的！如使用GNU Toolchain时，不能加载IntelOneAPI的环境变量；或是不能有奇特的 mpirun 存在于conda 包内等。

以GCCToolchain 默认设置为例，如此运行后，会做如下操作：

1．检查你的系统GCC 编译器版本（2025-02 版本 Toolchain新功能)，并链接GCC 编译器。输出类似于：

Checking system GCC version for gcc， intel and amd toolchain   
2 Your System gcc/g++/gfortran version should be consistent   
3 Minimum required version: 5   
4 Your gcc version: 11.4.0   
5 Your g++ version: 11.4.0   
6 Your gfortran version: 11.4.0   
7 Your GCC version seems to be enough for ABACUS installation.   
8 Compiling with 128 processes for target native.   
9 == ===== Finding GCC from system paths ======   
10 path to gcc is /usr/bin/gcc

11 path to g++ is /usr/bin/g++   
12 path to gfortran is /usr/bin/gfortran   
13 Found include directory /usr/include   
14 Step gcc took 0.00 seconds.   
15 Step intel took 0.00 seconds.   
16 Step amd took 0.00 seconds.

注：编译 ABACUS 及其相关依赖是有系统C++ 编译器最低版本要求的，至少要求支持C++11特性。同时，有一些依赖库还要求一部分的C++14特性，这都要求GCC版本至少不能小于5，即使是使用Intel 或AMD 相关依赖库和编译器进行编译安装时也是如此。然而，有一部分老旧超算依然维持着CentOS的内核和4.8.5版本的GCC，这些超算上的ABACUS 编译需要加载更高版本的GCC，或基于Toolchain编译安装GCC。用户可以通过gcc-version 命令检查自已的gcc 版本。

2.下载OpenBLAS，通过OpenBLAS 内置的 genarch 程序识别系统指令集和架构，在不同服务器上的不同输出如下

1 # Run toolchain on Intel-CPU Machine   
2 ==: ====== Getting proc arch info using OpenBLAS tools =====   
3 vget htps://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29-0 OpenBLAS-0.3.29.tar.gz --no-check   
4 --2025-05-06 16:58:40-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
5 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
6 Connecting to codeload.github.com (codeload.github.com)l20.205.243.165l:443... connected.   
7 HTTP request sent， awaiting response... 200 0K   
8 Length: unspecified [application/x-gzip]   
9 Saving to: ‘0penBLAS-0.3.29.tar.gz'   
10   
11 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 6.75MB/s in 3.5s   
12   
13 2025-05-06 16:58:45 (6.75 MB/s) - ‘0penBLAS-0.3.29.tar.gz' saved [24671913]   
14   
15 0penBLAS-0.3.29.tar.gz: OK   
16 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
17 OpenBLAS detected LIBCORE = skylakex   
18 OpenBLAS detected ARCH  = x86_64   
19   
20 # Run toolchain on AMD-CPU Machine   
21 == ====== Getting proc arch info using OpenBLAS tools ==   
22 wget htps://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29 -0 OpenBLAS-0.3.29.tar.gz --no-check   
23 -2025-05-06 04:49:02-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
24 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
25 Connecting to codeload.github.com (codeload.github.com)|20.205.243.165l:443... conected.   
26 HTTP request sent， awaiting response... 200 0K   
27 Length: unspecified [application/x-gzip]   
28 Saving to: ‘0penBLAS-0.3.29.tar.gz'   
29   
30 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 2.77MB/s in 10s   
31   
32 2025-05-06 04:49:13 (2.35 MB/s) - ‘0penBLAS-0.3.29.tar.gz’ saved [24671913]   
33

34 OpenBLAS-0.3.29.tar.gz: OK   
35 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
36 OpenBLAS detected LIBCORE = zen   
37 OpenBLAS detected ARCH = x86_64

3．自动下载并编译安装:编译套件CMake;并行库OpenMPI;数学库OpenBLAS,ScaLAPACK,FFTW,ELPA;泛函库LibXC;以及其他功能需要的库 CEREAL,RapidJSON 和其他可选项 LibRI,LibComm,LibTorch,Libnpy等。以AMD-7b12机器编译为例，完整输出如下：

MPI is detected and defaults to MPICH   
2 Checking system GCC version for gcc， intel and amd toolchain   
3 Your System gcc/g++/gfortran version should be consistent   
4 Minimum required version: 5   
5 Your gcc version: 11.4.0   
6 Your g++ version: 11.4.0   
7 Your gfortran version: 11.4.0   
8 Your GCC version seems to be enough for ABACUS installation.   
9 Compiling with 128 processes for target native.   
10 === ===== Finding GCC from system paths ====:   
11 path to gcc is /usr/bin/gcc   
12 path to g++ is /usr/bin/g++   
13 path to gfortran is /usr/bin/gfortran   
14 Found include directory /usr/include   
15 Step gcc took 0.00 seconds.   
16 Step intel took 0.00 seconds.   
17 Step amd took 0.00 seconds.   
18 == Getting proc arch info using OpenBLAS tools ==:   
19 wget htps://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29 -0 OpenBLAS-0.3.29.tar.gz --no-check   
20 --2025-05-06 04:49:02-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
21 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
22 Connecting to codeload.github.com (codeload.github.com)|20.205.243.165l:443... conected.   
23 HTTP request sent， awaiting response... 200 0K   
24 Length: unspecified [application/x-gzip]   
25 Saving to: ‘0penBLAS-0.3.29.tar.gz'   
26   
27 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 2.77MB/s in 10s   
28   
29 2025-05-06 04:49:13 (2.35 MB/s) - ‘0penBLAS-0.3.29.tar.gz’ saved [24671913]   
30   
31 OpenBLAS-0.3.29.tar.gz: OK   
32 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
33 OpenBLAS detected LIBCORE = zen   
34 OpenBLAS detected ARCH = x86_64   
35 ==================== Installing CMake ======== ====   
36 get htps://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.sh -0 cmake-3.31.7-linux-x86_64.sh --no-chec   
37 --2025-05-06 04:49:17-- https://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.sh   
38 Resolving cmake.org (cmake.org)... 66.194.253.25   
39 Connecting to cmake.org (cmake.org)l66.194.253.25l:443... connected.

40 HTTP request sent， awaiting response... 200 0K   
41 Length: 55005854 (52M) [text/x-sh]   
42 Saving to: ‘cmake-3.31.7-linux-x86_64.sh'   
43   
44 cmak-3.1.7-1ux-x8664. 10%[===========================>] 52.46M .2MB/s in 24s   
45   
46 2025-05-06 04:49:42 (2.23 MB/s) - ‘cmake-3.31.7-linux-x86_64.sh’ saved [55005854/55005854]   
47   
48 cmake-3.31.7-linux-x86_64.sh: 0K   
49 Checksum of cmake-3.31.7-linux-x86_64.sh 0k   
50 Installing from scratch into /home/mikoto/tests/tolchain-dev-test/abacus-develop-LTS/toolchain/install/cm   
51 Step cmake took 27.00 seconds.   
52 =================== Installing 0penMPI ========: =======sss   
53 vget htps://dowload.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2 -0 openmpi-5.0.7.tar.bz2 -   
54 --2025-05-06 04:49:44-- https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2   
55 Resolving dovnload.open-mpi.org (download.open-mpi.org)... 2600:900:2200:c000:16:d6ed:ec80:93a1，2600:900   
56 Connecting to download.open-mpi.0rg (download.open-mpi.org)l2600:9000:2200:c00:16:d6ed:ec80:93a1l:443..   
57 HTTP request sent， awaiting response... 200 0K   
58 Length: 30506871 (29M) [binary/octet-stream]   
59 Saving to: ‘openmpi-5.0.7.tar.bz2'   
60   
61 openmpi-5.0.7.tar.bz2 100%[============: ========>] 29.09M 33.0MB/s in 0.9s   
62   
63 2025-05-06 04:49:46 (33.0 MB/s) - ‘openmpi-5.0.7.tar.bz2' saved [30506871/30506871]   
64   
65 openmpi-5.0.7.tar.bz2: 0K   
66 Checksum of openmpi-5.0.7.tar.bz2 0k   
67 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/op   
68 Found directory /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi-5.0.7/b   
69 Found directory /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi-5.0.7/1   
70 Found directory /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi-5.0.7/i   
71 mpiexec is installed as /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi   
72 mpicc is instaled as /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/tolchain/install/opempi-5   
73 mpicxx is installed as /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/insta/openmpi-  
74 mpifort is installed as /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi   
75 Step openmpi took 324.00 seconds.   
76 ================ Insta11ing 0penLAS =================   
77 OpenBLAS-0.3.29.tar.gz is found   
78 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/op   
79 Installing OpenBLAS library for target ZEN   
80 Step openblas took 84.00 seconds.   
======= Installing LIBXC ========= ==s   
82 wget htps://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.bz2 -0 libxc-7.0.0.tar.bz2 --no-check   
83 --2025-05-06 04:56:32-- https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.bz2   
84 Resolving gitlab.com (gitlab.com)... 2606:4700:90:0:f22e:fbec:5bed:a9b9, 172.65.251.78   
85 Connecting to gitlab.com (gitlab.com)|2606:4700:90:0:f22e:fbec:5bed:a9b9l:443.. conected.   
86 HTTP request sent， awaiting response... 200 0K   
87 Length: unspecified [application/octet-stream]

88 Saving to: ‘libxc-7.0.0.tar.bz2'   
89   
90 libxc-7.0.0.tar.bz2 [ <=> ]49.98M 38.5MB/s in 1.3s   
91   
92 2025-05-06 04:56:34 (38.5 MB/s) - ‘libxc-7.0.0.tar.bz2’ saved [52408700]   
93   
94 libxc-7.0.0.tar.bz2: 0K   
95 Checksum of libxc-7.0.0.tar.bz2 0k   
96 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/li   
97 Step libxc took 49.00 seconds.   
98 ==================== Installing FFTW ===================   
99 wget http://www.fftw.org/fftw-3.3.10.tar.gz -0 fftw-3.3.10.tar.gz --no-check-certificate   
100 --2025-05-06 04:57:21-- http://ww.ftw.0rg/fftw-3.3.10.tar.gz   
101 Resolving www.ftw.org (www.fftw.org)... 216.137.187.110   
102 Connecting to ww.fftw.org (www.fftw.org)l216.137.187.110l:80... connected.   
103 HTTP request sent， awaiting response... 200 0K   
104 Length: 4144100 (4.0M) [application/octet-stream]   
105 Saving to: ‘ftw-3.3.10.tar.gz'   
106   
107 fftw-3.3.10.tar.gz 100%[=====: =====>] 3.95M  859KB/s in 4.7s   
108   
109 2025-05-06 04:57:27 (859 KB/s) - ‘fftw-3.3.10.tar.gz' saved [4144100/4144100]   
110   
111 fftw-3.3.10.tar.gz: OK   
112 Checksum of fftw-3.3.10.tar.gz 0k   
113 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/ff   
114 install float version fftw   
115 clean   
116 install double version fftw   
117 Step fftw took 142.00 seconds.   
118 =====: ======= Installing ScaLAPACK ======== ======   
119 wget https://codeload.github.com/Reference-ScaLAPACK/scalapack/tar.gz/v2.2.2 -0 scalapack-2.2.2.tar.gz.--   
120 --2025-05-06 04:59:43-- https://codeload.github.com/Reference-ScaLAPACK/scalapack/tar.gz/v2.2.2   
121 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
122 Connecting to codeload.github.com (codeload.github.com)l20.205.243.165l:443... connected.   
123 HTTP request sent， awaiting response... 200 0K   
124 Length: unspecified [application/x-gzip]   
125 Saving to: ‘scalapack-2.2.2.tar.gz'   
126   
127 scalapack-2.2.2.tar.gz [ <=> ]4.55M 1.64MB/s in 2.8s   
128   
129 2025-05-06 04:59:47 (1.64 MB/s) - ‘scalapack-2.2.2.tar.gz' saved [4769347]   
130   
131 scalapack-2.2.2.tar.gz: 0K   
132 Checksum of scalapack-2.2.2.tar.gz 0k   
13 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/sc   
134 Step scalapack took 12.00 seconds.   
135 ======: ===-====== Installing ELPA =============-=====

136 wget https:/elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.01.001/elpa-2025.01.001.tar.gz -0 e   
137 --2025-05-06 04:59:55-- htps://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.01.001/elpa-2025   
138 Resolving elpa.mpcdf.mpg.de (elpa.mpcdf.mpg.de)... 130.183.17.94   
139 Connecting to elpa.mpcdf.mpg.de (elpa.mpcdf.mpg.de)l130.183.17.94l:443... connected.   
140 HTTP request sent， awaiting response... 200 0K   
141 Length: 2169795 (2.1M) [application/gzip]   
142 Saving to: ‘elpa-2025.01.001.tar.gz'   
143   
144 e1p-225.01.001.targz 10%===========================>] 2.07m 1.8MB/s in 1.2s   
145   
146 2025-05-06 04:59:58 (1.68 MB/s) - ‘elpa-2025.01.001.tar.gz’ saved [2169795/2169795]   
147   
148 elpa-2025.01.001.tar.gz: OK   
149 Checksum of elpa-2025.01.001.tar.gz 0k   
150 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/instal/el   
151 Step elpa took 121.00 seconds.   
152 ==================== Instaling CEReAL ==================   
153 ==> Notice: This version of CEREAL is downloaded in GitHub master repository <===   
154 wget https://odeload.github.com/USCiLab/cereal/tar.gz/master -0 cereal-master.tar.gz --no-check-certific   
155 --2025-05-06 05:01:56-- https://codeload.github.com/UsCiLab/cereal/tar.gz/master   
156 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
157 Connecting to codeload.github.com (codeload.github.com)l20.205.243.165l:443... connected.   
158 HTTP request sent， awaiting response... 200 0K   
159 Length: unspecified [application/x-gzip]   
160 Saving to: ‘cereal-master.tar.gz'   
161   
162 cereal-master.tar.gz [<=> ]377.35K381KB/s in 1.0s   
163   
164 2025-05-06 05:01:59 (381 KB/s) - ‘cereal-master.tar.gz' saved [386409]   
165   
166 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/ce   
167 Step cereal took 3.00 seconds.   
168 ==: ======= Installing RAPIDJSON ===================   
169 ===> Notice: This version of rapidjson is downloaded in GitHub master repository <===   
170 vget https://codeload.github.com/Tencent/rapidjson/tar.gz/master -0 rapidjson-master.tar.gz --no-check-ce   
171 --2025-05-06 05:01:59-- https://codeload.github.com/Tencent/rapidjson/tar.gz/master   
172 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
173 Connecting to codeload.github.com (codeload.github.com)l20.205.243.165l:443... conected.   
174 HTTP request sent， awaiting response... 200 0K   
175 Length: unspecified [application/x-gzip]   
176 Saving to: ‘rapidjson-master.tar.gz'   
177   
178 rapidjson-master.tar.gz [<=> ]1.06M919KB/s in 1.2s   
179   
180 2025-05-06 05:02:01 (919 KB/s) - ‘rapidjson-master.tar.gz' saved [1116059]   
181   
182 Installing from scratch into /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/ra   
183 Step rapidjson took 2.00 seconds.

184 Step libtorch took 0.00 seconds.   
185 Step libnpy took 0.00 seconds.   
186 Step libri took 0.00 seconds.   
187 Step libcomm took 0.00 seconds.   
188 ====: ===== usage ==   
189 Done!   
190 To use the installed tools and libraries and ABACUS version   
191 compiled with it you will first need to execute at the prompt:   
192 source /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/setup   
193 To build ABACUS by gnu-toolchain, just use:   
194 ./build_abacus_gnu.sh   
195 To build ABACUS by intel-toolchain, just use:   
196 ./build_abacus_intel.sh   
197 To build ABACUS by amd-toolchain in gcc-aocl, just use:   
198 ./build_abacus_gnu-aocl.sh   
199 or you can modify the builder scripts to suit your needs.

编译成功后，根据输出提示，运行build_abacus_gnu.sh

1 sh ./build_abacus_gnu.sh

即可完成ABACUS本体的编译安装，大致输出如下：

```diff
-- The CXX compiler identification is GNU 11.4.0
2 -- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
4 -- Check for working Cxx compiler: /usr/bin/g++ - skipped
-- Detecting CXx compile features
-- Detecting CXx compile features - done
-- RapidJSON found. Headers:
-- Found Git: /usr/bin/git (found version "2.34.1")
9 -- Found git: attempting to get commit info...
10 fatal: not a git repository (or any of the parent directories): .git
1 fatal: not a git repository (or any of the parent directories): .git
12 CMake Warning at CMakeLists.txt:104 (message):
13 Failed to get git commit info
14 --Found Cereal: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/instal/cereal-master/inclu
15 -- Found PkgConfig: /usr/bin/pkg-config (found version "0.29.2")
16 --Found ELPA: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/elpa-2025.01.001/cpu/:
17 -- Performing Test ELPA_VERSION_SATISFIES
18 -- Performing Test ELPA_VERSION_SATISFIES - Success
19 --Found MPI_CXx: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openmpi-5.0.7/lib/:
20 -- Found MPI: TRUE (found version "3.1")
21 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
22 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - SuccesS
23 -- Found Threads: TRUE
24 -- Found OpenMP_Cxx: -fopenmp (found version "4.5")
25 -- Found OpenMP: TRUE (found version "4.5")
```

26 Looking for a CUDA compiler   
27 -- Looking for a CUDA compiler - NOTFOUND   
28 -- Found FTW3: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/fft-3.3.10/lib/ibf:   
29 -- Looking for sgemm_   
30 -- Looking for sgemm_ - not found   
31 -- Looking for sgemm_   
32 -- Looking for sgemm_ - found   
33 -- Found BLAS: home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/openblas-0.3.29/lib/1:   
34 -- Looking for cheev_   
35 -- Looking for cheev_ - found   
36 --Found LAPACK: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/instal/openblas-0.3.29/lib   
37 CMake Warning (dev) at CMakeLists.txt:439 (find_package):   
38 Policy CMP0144 is not set: find_package uses upper-case <PACKAGENAME>_R00T   
39 variables. Run "cmake --help-policy CMP0144" for policy details. Use the   
40 cmake_policy command to set the policy and suppress this warning.   
41   
42 Environment variable SCALAPACK_ROOT is set to:   
43   
44 /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/scalapack-2.2.2   
45   
46 For compatibility， find_package is ignoring the variable, but code in a   
47 .cmake module might still use it.   
48 This warning is for project developers. Use -Wno-dev to suppress it.   
49   
50 -- Found ScaLAPACK: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/scalapack-2.2.2/:   
51 -- Checking for one of the modules 'libxc'   
52 -- Found Libxc: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/install/libxc-7.0.0/lib/libx   
53 -- Found Libxc: version 7.0.0   
54 -- Configuring done (1.4s)   
55 -- Generating done (0.2s)   
56 -- Build files have been vriten to: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/build_abacus_gu   
57 [0%] Building CXx object source/module_rdmft/CMakeFiles/rdmft.dir/rdmft.cpp.o   
58 [0%] Building Cxx object source/CMakeFiles/device.dir/module_hamilt_pw/hamilt_pudft/kernels/nonlocal_op.cpp.0   
59 ：   
60 [100%] Building Cxx object source/module_base/CMakeFiles/base.dir/module_mixing/pulay_mixing.cpp.0   
61 [100%] Building Cxx object source/module_base/CMakeFiles/base.dir/module_mixing/broyden_mixing.cpp.0   
62 [100%] Built target io_input   
63 [100%] Built target io_advanced   
64 [100%] Built target cell   
65 [100%] Built target hamilt_lcao   
66 [100%] Built target operator_ks_lcao   
67 [100%] Built target base   
68 [100%] Built target esolver   
69 [100%] Built target io_basic   
70 [100%] Built target lr   
71 [100%] Building Cxx object CMakeFiles/abacus.dir/source/main.cpp.0   
72 [100%] Linking CXx executable abacus   
73 [100%] Built target abacus

74 -- Install configuration: ""   
75 -- Installing: /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/bin/abacus   
76 == usage ==   
77 Done!   
78 To use the installed ABACUS version   
79 You need to source /home/mikoto/tests/toolchain-dev-test/abacus-develop-LTS/toolchain/abacus_env.sh first !

如上输出即为完成了ABACUS 的编译安装。若想使用如此编译安装的 ABACUS，只需要在Toolchain目录下加载自动生成的环境变量脚本：

1 source abacus_env.sh

即可加载 ABACUS 所有依赖库环境和 ABACUS 本体，此时可以通过 abacus -version 命令确认ABACUS 已正确安装和加载：

1 (base) mikoto@amd-cpu:toolchain\$ source abacus_env.sh   
2 (base) mikoto@amd-cpu:toolchain\$ abacus --version   
3 ABACUS version v3.10.0

## 2.5 Toolchain 编译 ABACUS 教程(2)-Intel

## 2.5.1 简介

此前的教程介绍如何使用GCC Toolchain 简单直接地从头编译ABACUS，这一方案兼容性最好，但编译所得的ABACUS 不一定最高效，尤其对于很多配置了相应IntelOneAPI套件的Intel-CPU服务器。本教程则将侧重于如何利用Intel Toolchain，让编译得到的ABACUS 获得更高性能。与此同时，本教程还会介绍如何编辑Toolchain的关键脚本，达到切换不同编译依赖组件或是加入特定功能插件的效果。

## 2.5.2 基于Intel Toolchain 安装 ABACUS

下面我们将以Intel Toolchain 为例，展示安装 LTS-3.10.0版本 ABACUS 的过程。其中，针对不同的服务器特点，我们需要针对性地调整Intel Tolchain 的各个脚本，教程作者习惯使用vim 进行调整，会基于vim 教学,具体操作可以采用任何在Linux 服务器上编辑文件的方法。

注意：使用 Toolchain进行源码安装前，务必确保你正确加载了对应的IntelOneAPI环境或其他相关环境变量！

```shell
1 # load intel-oneapi environment via source
2 source /path/to/intel/oneapi/setvars.sh
3 # load intel-oneapi environment via module-env if exists
4 module load mkl mpi compiler
```

## 在IntelOneAPI2024+依赖下直接安装ABACUS

如果你的服务器IntelOneAPI版本在2024.0版本以上，一般来说，你可以直接通过下述步骤进行ABACUS的编译安装。（202502版本 Toolchain 已通过拉取最新版本依赖库方式支持了OneAPI2025+版本）

1.下载ABACUS仓库。可以通过gitclone下载，这一做法允许用户通过git包管理器快速更新或切换版本。也可以通过 wget拉取ABACUS 仓库压缩包，这样得到的 ABACUS 仓库没有git相关的信息，文件体积更

小。

1 # via github   
2 git clone https://github.com/deepmodeling/abacus-develop.git -b LTS   
3 # via wget from github by codeload: can be used under CN Internet   
4 wget https://codeload.github.com/deepmodeling/abacus-develop/tar.gz/LTS -0 abacus-LTS.tar.gz   
5 tar -zxvf abacus-LTS.tar.gz

## 2．进入toolchain目录

1 cd abacus-develop/toolchain   
2 # if you download abacus via wget from codeload   
3 cd abacus-develop-LTS/toolchain

## 3．运行 toolchain_intel.sh 脚本

1 sh ./toolchain_intel.sh

如此即可开始基于 Toolchain的ABACUS 依赖库编译安装。以Intel Toolchain 默认设置为例，会做如下操作：1．检查你的系统GNU编译器版本（2025-02版本Toolchain新功能），并链接Intel编译器，输出类似于：

1 MPI is detected and it appears to be Intel MPI   
2 Checking system GCC version for gcc, intel and amd toolchain   
3 Your System gcc/g++/gfortran version should be consistent   
4 Minimum required version: 5   
5 Your gcc version: 11.3.0   
6 Your g++ version: 11.3.0   
7 Your gfortran version: 11.3.0   
8 Your GCC version seems to be enough for ABACUS installation.   
Using MKL， so openblas is disabled.   
10 Compiling with 16 processes for target native.   
11 Step gcc took 0.00 seconds.   
12 ==: ==== Finding Intel compiler from system paths   
13 path to icx is /opt/intel/oneapi/compiler/2025.1/bin/icx   
14 path to icpx is /opt/intel/oneapi/compiler/2025.1/bin/icpx   
15 path to ifx is /opt/intel/oneapi/compiler/2025.1/bin/ifx   
16 CC is /opt/intel/oneapi/compiler/2025.1/bin/icx   
17 CXX is /opt/intel/oneapi/compiler/2025.1/bin/icpx   
18 FC is /opt/intel/oneapi/compiler/2025.1/bin/ifx   
19 Step intel took 0.00 seconds.   
20 Step amd took 0.00 seconds.

注：编译ABACUS 及其相关依赖是有系统C++编译器最低版本要求的，GCC版本至少不能小于5，即使是使用Intel或AMD相关依赖库和编译器进行编译安装时也是如此。然而，有一部分老旧超算依然维持着CentOS 的内核和4.8.5版本的GCC，因而如果在Intel Toolchain 编译过程中出现奇特报错，一大可能就是服务器GCC版本过低，使得Intel编译器不能正常编译ABACUS 相关程序，此时请联系服务器管理员。

2.下载OpenBLAS（即使 Intel Toolchain 也会进行这一操作)，通过 OpenBLAS 内置的 genarch 程序识别系统指令集和架构，在服务器上的输出如下：

1 # Run toolchain on Intel-CPU Machine   
==== Getting proc arch info using OpenBLAS tools =====:   
3 wget https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29 -0 OpenBLAS-0.3.29.tar.gz--no-check   
1 --2025-05-06 16:58:40-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
5 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
6 Connecting to codeload.github.com (codeload.github.com)|20.205.243.165l:443... connected.   
HTTP request sent， awaiting response... 200 0K   
8 Length: unspecified [application/x-gzip]   
9 Saving to: ‘0penBLAS-0.3.29.tar.gz'   
10   
11 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 6.75MB/s in 3.5s   
12   
13 2025-05-06 16:58:45 (6.75 MB/s) - ‘0penBLAS-0.3.29.tar.gz’ saved [24671913]   
14   
15 0penBLAS-0.3.29.tar.gz: OK   
16 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
17 OpenBLAS detected LIBCORE = skylakex   
18 OpenBLAS detected ARCH = x86_64

## 3．自动下载CMake 编译库.

4．从环境变量中定位并链接IntelMPI并行库和IntelMKL数学库

== Finding Intel MPI from system paths ==   
2 path to mpiexec is /opt/intel/oneapi/mpi/2021.15/bin/mpiexec   
3 path to mpiicx is /opt/intel/oneapi/mpi/2021.15/bin/mpiicx   
4 path to mpiicpx is /opt/intel/oneapi/mpi/2021.15/bin/mpiicpx   
5 path to mpiifx is /opt/intel/oneapi/mpi/2021.15/bin/mpiifx   
6 Found lib directory /opt/intel/oneapi/mpi/2021.15/lib   
7 libmpi is found in ld search path   
8 libmpicxx is found in ld search path   
9 I_MPI_CXX is icpx   
10 I_MPI_CC is icx   
11 I_MPI_FC is ifx   
12 MPICXX is /opt/intel/oneapi/mpi/2021.15/bin/mpiicpx   
13 MPICC is /opt/intel/oneapi/mpi/2021.15/bin/mpiicx   
14 MPIFC is /opt/intel/oneapi/mpi/2021.15/bin/mpiifx   
15 Step intelmpi took 0.00 seconds.   
16 = ==== Finding MKL from system paths ====:   
17 MKLR00T is found to be /opt/intel/oneapi/mkl/2025.1   
18 libm is found in ld search path   
19 libdl is found in ld search path   
20 Step mkl took 0.00 seconds.

5．自动下载并编译安装：数学库 ELPA，泛函库LibXC，其他功能库CEREAL,RapidJSON以及可选功能库：LibRI,LibComm,LibTorch,Libnpy 等。完整输出如下：（本案例基于Bohr容器完成）

1 MPI is detected and it appears to be Intel MPI   
2 Checking system GCC version for gcc， intel and amd toolchain

3 Your System gcc/g++/gfortran version should be consistent   
4 Minimum required version: 5   
5 Your gcc version: 11.3.0   
6 Your g++ version: 11.3.0   
7 Your gfortran version: 11.3.0   
8 Your GCC version seems to be enough for ABACUS installation.   
9 Using MKL， so openblas is disabled.   
10 Compiling with 16 processes for target native.   
11 Step gcc took 0.00 seconds.   
12 ==================== Finding Intel compiler from system paths ========:   
13 path to icx is /opt/intel/oneapi/compiler/2025.1/bin/icx   
14 path to icpx is /opt/intel/oneapi/compiler/2025.1/bin/icpx   
15 path to ifx is /opt/intel/oneapi/compiler/2025.1/bin/ifx   
16 CC is /opt/intel/oneapi/compiler/2025.1/bin/icx   
17 CXX is /opt/intel/oneapi/compiler/2025.1/bin/icpx   
18 FC is /opt/intel/oneapi/compiler/2025.1/bin/ifx   
19 Step intel took 0.00 seconds.   
20 Step amd took 0.00 seconds.   
21 ======: ======= Getting proc arch info using OpenBLAS tools ===   
22 vget https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29 -0 OpenBLAS-0.3.29.tar.gz --no-check   
23 --2025-05-09 17:15:04-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
24 Resolving ga.dp.tech (ga.dp.tech)... 10.255.254.18, 10.255.254.37， 10.255.254.7   
25 Connecting to ga.dp.tech (ga.dp.tech)l10.255.254.18l:8118... connected.   
26 Proxy request sent， awaiting response... 200 0K   
27 Length: unspecified [application/x-gzip]   
28 Saving to: '0penBLAS-0.3.29.tar.gz'   
29   
30 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 4.64MB/s in 5.5s   
31   
32 2025-05-09 17:15:11 (4.30 MB/s) - '0penBLAS-0.3.29.tar.gz' saved [24671913]   
33   
34 OpenBLAS-0.3.29.tar.gz: OK   
35 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
36 ./f_check: 100: [: Illegal number:   
37 OpenBLAS detected LIBCORE = skylakex   
38 OpenBLAS detected ARCH  = x86_64   
39 == Installing CMake =:   
40 wgethtps://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.sh -0 cmake-3.31.7-linux-x86_64.sh --no-chec   
41 --2025-05-09 17:15:16-- https://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.sh   
42 Resolving ga.dp.tech (ga.dp.tech)... 10.255.254.37， 10.255.254.7， 10.255.254.18   
43 Connecting to ga.dp.tech (ga.dp.tech)l10.255.254.37l:8118... connected.   
44 Proxy request sent， awaiting response... 200 0K   
45 Length: 55005854 (52M) [text/x-sh]   
46 Saving to: 'cmake-3.31.7-linux-x86_64.sh'   
47   
48 cmake-3.31.7-linux-x86_64. 100%[== ======>] 52.46M 7.10MB/s in 8.7s   
49   
50 2025-05-09 17:15:25 (6.04 MB/s) - 'cmake-3.31.7-linux-x86_64.sh' saved [55005854/55005854]

---

## 第 41–60 页

51   
52 cmake-3.31.7-linux-x86_64.sh: 0K   
53 Checksum of cmake-3.31.7-linux-x86_64.sh 0k   
54 Installing from scratch into /opt/abacus-develop-LTS/toolchain/install/cmake-3.31.7   
55 Step cmake took 11.00 seconds.   
56 =================== Finding Intel MPI from system pths ===================   
57 path to mpiexec is /opt/intel/oneapi/mpi/2021.15/bin/mpiexec   
58 path to mpiicx is /opt/intel/oneapi/mpi/2021.15/bin/mpiicx   
59 path to mpiicpx is /opt/intel/oneapi/mpi/2021.15/bin/mpiicpx   
60 path to mpiifx is /opt/intel/oneapi/mpi/2021.15/bin/mpiifx   
61 Found lib directory /opt/intel/oneapi/mpi/2021.15/lib   
62 libmpi is found in ld search path   
63 libmpicxx is found in ld search path   
64 I_MPI_CXX is icpx   
65 I_MPI_CC is icx   
66 I_MPI_FC is ifx   
67 MPICXX is /opt/intel/oneapi/mpi/2021.15/bin/mpiicpx   
68 MPICC is /opt/intel/oneapi/mpi/2021.15/bin/mpiicx   
69 MPIFC is /opt/intel/oneapi/mpi/2021.15/bin/mpiifx   
70 Step intelmpi took 0.00 seconds.   
71 =================== Finding ML from system paths ==================   
72 MKLR0OT is found to be /opt/intel/oneapi/mkl/2025.1   
73 libm is found in ld search path   
74 libdl is found in ld search path   
75 Step mkl took 0.00 seconds.   
76 ==========: ======== Instaling IbxC ==================   
77 vget htps://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.bz2 -0 libxc-7.0.0.tar.bz2 --no-check   
78 --2025-05-09 17:15:27-- https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.bz2   
79 Resolving ga.dp.tech (ga.dp.tech)... 10.255.254.18，10.255.254.7，10.255.254.37   
80 Connecting to ga.dp.tech (ga.dp.tech)l10.255.254.18l:8118... connected.   
81 Proxy request sent， awaiting response... 200 0K   
82 Length: unspecified [application/octet-stream]   
83 Saving to: 'libxc-7.0.0.tar.bz2'   
84   
85 libxc-7.0.0.tar.bz2 [ <=> ] 49.98M 7.63MB/s in 7.5s   
86   
87 2025-05-09 17:15:36 (6.68 MB/s) - '1ibxc-7.0.0.tar.bz2' saved [52408700]   
88   
89 libxc-7.0.0.tar.bz2: 0K   
90 Checksum of libxc-7.0.0.tar.bz2 0k   
91 Installing from scratch into /opt/abacus-develop-LTS/toolchain/install/libxc-7.0.0   
92 Step libxc took 42.00 seconds.   
93 Step fftw took 0.00 seconds.   
94 Step scalapack took 0.00 seconds.   
95 =================== Instal1ing ELPA ===================   
96 vget https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.01.001/elpa-2025.01.001.tar.gz -0 e   
97 --2025-05-09 17:16:09-- https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.01.001/elpa-2025   
98 Resolving ga.dp.tech (ga.dp.tech)... 10.255.254.18， 10.255.254.7， 10.255.254.37

147 Step libcomm took O.00 seconds.   
148 ==== ===== usage =====:   
149 Done!   
150 To use the installed tools and libraries and ABACUS version   
151 compiled with it you will first need to execute at the prompt:   
152 source /opt/abacus-develop-LTS/toolchain/install/setup   
153 To build ABACUS by gnu-toolchain， just use:   
154 ./build_abacus_gnu.sh   
155 To build ABACUS by intel-toolchain， just use:   
156 ./build_abacus_intel.sh   
157 To build ABACUS by amd-toolchain in gcc-aocl， just use:   
158 ./build_abacus_gcc-aocl.sh   
159 To build ABACUS by amd-toolchain in aocc-aocl， just use:   
160 ./build_abacus_aocc-aocl.sh   
161 or you can modify the builder scripts to suit your needs.

如此编译依赖库成功后，根据输出提示，运行build_abacus_intel.sh

1 sh ./build_abacus_intel.sh

即可完成ABACUS 本体的编译安装，大致输出如下：

```vhdl
-- The CXX compiler identification is IntelLLVM 2025.1.1
-- Detecting CXX compiler ABI info
3 -- Detecting CXX compiler ABI info - done
-- Check for working CXx compiler: /opt/intel/oneapi/compiler/2025.1/bin/icpx - skipped
5 -- Detecting CXx compile features
-- Detecting CXX compile features - done
-- RapidJSON found. Headers:
8 -- Found Git: /usr/bin/git (found version "2.34.1")
9 -- Found git: attempting to get commit info...
10 fatal: not a git repository (or any of the parent directories): .git
11 fatal: not a git repository (or any of the parent directories): .git
12 CMake Warning at CMakeLists.txt:104 (message):
13 Failed to get git commit info
14
15
16 -- Found Cereal: /opt/abacus-develop-LTS/toolchain/install/cereal-master/include/cereal
17 -- Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
18 -- ELPA : We need pkg-config to get all information about the elpa library
19 -- Found ELPA: /opt/abacus-develop-LTS/toolchain/install/elpa-2025.01.001/cpu/lib/libelpa_openmp.so
20 -- Performing Test ELPA_VERSION_SATISFIES
21 -- Performing Test ELPA_VERSION_SATISFIES - SuccesS
22 -- Found MPI_CxX: /opt/intel/oneapi/mpi/2021.15/lib/libmpicxx.so (found version "3.1")
23 -- Found MPI: TRUE (found version "3.1")
24 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
25 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - SuccesS
26 -- Found Threads: TRUE
```

```diff
27 -- Found OpenMP_Cxx: -fiopenmp (found version "5.1")
28 -- Found OpenMP: TRUE (found version "5.1")
29 -- Looking for a CUDA compiler
30 -- Looking for a CUDA compiler - NOTFOUND
31 -- MKL_VERSION: 2025.1.0
32 -- MKL_R00T: /opt/intel/oneapi/mkl/2025.1
33 -- MKL_ARCH: intel64
34 -- MKL_SYCL_LINK: None, set to ‘ dynamic' by default
35 -- MKL_LINK: None， set to ‘ dynamic` by default
36 MKL_SYCL_INTERFACE_FULL: intel_lp64
MKL_INTERFACE_FULL: intel_1p64
38 -- MKL_SYCL_THREADING: None， set to ‘ tbb_thread' by default
39 -- MKL_THREADING: None， set to ‘ intel_thread' by default
40 MKL_MPI: None, set to ‘ intelmpi’ by default
41 -- Experimental oneMKL Data Fitting SYCL API does not support LP64 on CPU
42 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_scalapack_lp64.so
43 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_cdft_core.so
44 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_intel_lp64.so
45 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_intel_thread.so
46 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_core.so
47 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_blacs_intelmpi_lp64.so
48 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_blas.so
49 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_lapack.so
50 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_dft.so
51 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_sparse.so
52 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_data_fitting.so
53 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_rng.so
54 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_stats.so
55 Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_sycl_vm.so
56 -- Found /opt/intel/oneapi/mkl/2025.1/lib/libmkl_tbb_thread.so
57 -- Found /opt/intel/oneapi/compiler/2025.1/lib/libiomp5.so
58 -- Could NOT find PkgConfig (missing: PKG_CONFIG_EXECUTABLE)
59 -- Found Libxc: version 7.0.0
60 -- Configuring done (1.3s)
61 -- Generating done (0.2s)
62 -- Build files have been written to: /opt/abacus-develop-LTS/build_abacus_intel
63 [ 0%] Building Cxx object source/CMakeFiles/driver.dir/driver.cpp.0
64
65 [100%] Built target io_basic
66 [100%] Building Cxx object CMakeFiles/abacus.dir/source/main.cpp.0
67 [100%] Linking CXx executable abacus
68 [100%] Built target abacus
69 -- Install configuration: ""
70 -- Installing: /opt/abacus-develop-LTS/bin/abacus
71 ======================== usage ========== ======ss
72 Done!
73 To use the installed ABACUS version
74 You need to source /opt/abacus-develop-LTS/toolchain/abacus_env.sh first !
```

如上输出即为完成了ABACUS 的编译安装。注意，从系统内读人的环境变量并不会被写人到abacus_env.sh对应的 instal/setup 文件中，因此在实际使用这一abacus 时，除了 source abacus_env.sh，你还需要加载对应的Intel-OneAPI环境。

1 # after load intel-OneAPI env via source or module-load   
2 source abacus_env.sh

如此即可加载 ABACUS 所有依赖库环境和 ABACUS 本体，此时可以通过 abacus-version命令确认ABACUS已正确安装和加载：

1 root@bohrium:toolchain# source abacus_env.sh   
2 root@bohrium:toolchain# abacus --version   
3 ABACUS version v3.10.0

## 基于较旧的Intel-OneAPI安装ABACUS

很多情况下，我们被迫使用旧版本的OneAPI，如

·使用Intel库编译GPU支持的ABACUS（此时只能用 icpc 编译器编译 ABACUS）。

·在AMD-CPU服务器上使用Intel-OneAPI编译ABACUS（此时只能用icpc 编译器编译 ELPA）。需要注意的是，在AMD-CPU 服务器上使用Intel-OneAPI编译的 ABACUS，其计算性能将较其他方案偏低。

·服务器上仅有旧版的OneAPI（但最好在 2023 版及以上）或 parallel_xe_studio。

此时，我们可以通过编辑关键脚本来完成基于 Toolchain 的 ABACUS 编译，用户层面操作的关键脚本包括 toolchain_\*.sh 和build_abacus_\*.sh。对于 Intel Toolchain 而言，相应脚本则是 toolchain_intel.sh 和build_abacus_intel.sh。

通过vim打开 toolchain_intel.sh

```shell
1 #!/bin/bash
2 #SBATCH -J install
3 #SBATCH -N 1
4 #SBATCH -n 16
5 #SBATCH -o compile.log
6 #SBATCH -e compile.err
7
8 # JamesMisaka in 2025-05-05
9 # install abacus dependency by intel-toolchain
10 # use mkl and intelmpi
11 # but mpich and openmpi can also be tried
12 # libtorch and libnpy are for deepks support，which can be =no
13 # gpu-lcao supporting modify: CUDA_PATH and --enable-cuda
14 # export CUDA_PATH=/usr/local/cuda
15
16 # module load mkl mpi compiler
17
18 ./install_abacus_toolchain.sh \
19 --with-intel=system \
20 --math-mode=mkl \
21 --with-gcc=no \
22 --with-intelmpi=system \
```

```shell
23 --with-cmake=install \
24 --with-scalapack=no \
25 --with-libxc=install \
26 --with-fftw=no \
27 --with-elpa=install \
28 --with-cereal=install \
29 --with-rapidjson=install \
30 --with-libtorch=no \
31 --with-libnpy=no \
32 --with-libri=no \
33 --with-libcomm=no \
34 --with-intel-classic=no \
35 丨 tee compile.log
36 # for using AMD-CPU or GPU-version: set --with-intel-classic=yes
37 # to enable gpu-lcao， add the following lines:
38 # --enable-cuda \
39 # --gpu-ver=75 \
40 # one should check your gpu compute capability number
```

输人:34切到第34行，按i键或a键进人编辑模式（好奇的小伙伴可以观察二者的区别），将-with-intel-classic=no 修改为-with-intel-classic=yes,呈现效果为：

```shell
1 ./install_abacus_toolchain.sh \
2 --with-intel=system \
3 --math-mode=mkl \
4 --with-gcc=no \
5 --with-intelmpi=system \
6 --with-cmake=install \
7 --with-scalapack=no \
8 --with-libxc=install \
9 --with-fftw=no \
10 --with-elpa=install \
11 --with-cereal=install \
12 --with-rapidjson=install \
13 --with-libtorch=no \
14 --with-libnpy=no \
15 --with-libri=no \
16 --with-libcomm=no \
17 --with-intel-classic=yes \
18 l tee compile.log
```

可以发现 toolchain\*.sh脚本实际上是通过调用主脚本 install_abacus_toolchain.sh来完成编译的，只是针对不同的toolchain进行了不同的编译选项设置，方便用户开箱即用和简单编辑。需要注意的，如此使用bash 命令时，各个选项其实是连接在一起的，也即分行符号”后"面不能有任何空格或其他字符，并且各个分行之间不能有注释。

按一下退出键ESC 退出编辑模式，输人:wq保存并退出。如此这般，Toolchain 将在运行时识别传统 Intel 编译器（ic,icpc,ifort）及其MPI库进行依赖库编译，运行 toolchain_intel.sh时，会有与此前不同的输出：

==== ========= Finding Intel compiler from system paths =======:   
2 path to icc is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/icc   
3 path to icpc is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/icpc   
4 path to ifort is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/ifort   
5 CC is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/icc   
6 CXx is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/icpc   
FC is /mnt/sg001/opt/intel/oneapi/compiler/2022.2.1/linux/bin/intel64/ifort   
8 Step intel took 0.00 seconds.   
9 Step amd took 0.00 seconds.   
10 ==: ======= Getting proc arch info using OpenBLAS tools ====   
11 OpenBLAS detected LIBCORE = skylakex   
12 OpenBLAS detected ARCH = x86_64   
13 ==================== Finding CMake from system paths =====\* ===   
14 path to cmake is /mnt/sg001/home/fz_pku_jh/software/cmake/3.31.7/bin/cmake   
15 Step cmake took 0.00 seconds.   
16 === ===== Finding Intel MPI from system paths =====   
17 path to mpiexec is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiexec   
18 path to mpiicc is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiicc   
19 path to mpiicpc is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiicpc   
20 path to mpiifort is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiifort   
21 Found lib directory /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/lib/release   
22 libmpi is found in ld search path   
23 libmpicxx is found in ld search path   
24 I_MPI_CXX is icpc   
25 I_MPI_CC is icc   
26 I_MPI_FC is ifort   
27 MPICXX is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiicpc   
28 MPICC is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiicc   
29 MPIFC is /mnt/sg001/opt/intel/oneapi/mpi/2021.7.1/bin/mpiifort   
30 Step intelmpi took 1.00 seconds.   
31 ====: ======= Finding MKL from system paths ====   
32 MKLR00T is found to be /mnt/sg001/opt/intel/oneapi/mkl/2022.2.1   
33 libm is found in ld search path   
34 libdl is found in ld search path   
35 Step mkl took 0.00 seconds.

读者可思考：为什么上面的编译输出中，cmake 是来自于system的？我还修改了哪个编译选项？这个问题的答案将在下一部分揭露。

如此编辑并运行完 toolchain_intel.sh后，进一步通过 vim 打开 build_abacus_intel.sh

```shell
1 #!/bin/bash
2 #SBATCH -J build
3 #SBATCH -N 1
4 #SBATCH -n 16
5 #SBATCH -o install.log
6 #SBATCH -e install.err
7 # JamesMisaka in 2025.03.09
```

```shell
9 # Build ABACUS by intel-toolchain
10
11 # module load mkl compiler mpi
12 # source path/to/setvars.sh
13
14 ABACUS_DIR=..
15 T00L=$(pwd)
16 INSTALL_DIR=$TOOL/install
17 Source $INSTALL_DIR/setup
18 cd $ABACUS_DIR
19 ABACUS_DIR=$(pwd)
20
21 BUILD_DIR=build_abacus_intel
22 rm -rf $BUILD_DIR
23
24 PREFIX=$ABACUS_DIR
25 ELPA=$INSTALL_DIR/elpa-2025.01.001/cpu
26 # ELPA=$INSTALL_DIR/elpa-2025.01.001/nvidia # for gpu-lcao
27 CEREAL=$INSTALL_DIR/cereal-master/include/cereal
28 LIBXC=$INSTALL_DIR/libxC-7.0.0
29 RAPIDJSON=$INSTALL_DIR/rapidjson-master/
30 # LIBTORCH=$INSTALL_DIR/libtorch-2.1.2/share/cmake/Torch
31 # LIBNPY=$INSTALL_DIR/libnpy-1.0.1/include
32 # LIBRI=$INSTALL_DIR/LibRI-O.2.1.0
33 # LIBCOMM=$INSTALL_DIR/LibComm-master
34 # DEEPMD=$H0ME/apps/anaconda3/envs/deepmd # v3.0 might have problem
35
36 # Notice: if you are compiling With AMD-CPU or GPU-version ABACUS, then ‘icpc' and ‘mpiicpc` compil(
37 cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
38 -DCMAKE_CXX_COMPILER=icpx \
39 -DMPI_CXX_COMPILER=mpiicpx \
40 -DMKLROOT=$MKLROOT \
41 -DELPA_DIR=$ELPA \
42 -DCEREAL_INCLUDE_DIR=$CEREAL \
43 -DLibxc_DIR=$LIBXC \
44 -DENABLE_LCAO=ON \
45 -DENABLE_LIBXC=ON \
46 -DUSE_OPENMP=ON\
47 -DUSE_ELPA=ON \
48 -DENABLE_RAPIDJSON=ON \
49 -DRapidJSON_DIR=$RAPIDJSON \
50# -DENABLE_DEEPKS=1\
51 # -DTorch_DIR=$LIBTORCH \
52 # -Dlibnpy_INCLUDE_DIR=$LIBNPY \
53# -DENABLE_LIBRI=ON \
54 # -DLIBRI_DIR=$LIBRI \
55# -DLIBCOMM_DIR=$LIBCOMM \
56# -DDeePMD_DIR=$DEEPMD \
```

```shell
57 # -DUSE_CUDA=ON \
58 # -DENABLE_CUSOLVERMP=ON\
59# -D CAL_CUS0LVERMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2x.xx/math_libs/1x.x/targets/x86_64
60
61 cmake --build $BUILD_DIR -j ‘nproc"
62 cmake --install $BUILD_DIR 2>/dev/null
63
64 # if one want's to include deepmd, your system gcc version should be >= 11.3.0 for glibc requirements
65
66 # generate abacus_env.sh
67 cat << EOF > "${TOOL}/abacus_env.sh"
68 #!/bin/bash
69 source $INSTALL_DIR/setup
70 export PATH="${PREFIX}/bin":\${PATH}
71 EOF
72
73 # generate information
74 cat << EOF
75 ===== === usage ==
76 Done!
77 To use the installed ABACUS version
78 You need to source ${TooL}/abacus_env.sh first !
79 ""
80 E0F
```

修改ABACUS 编译所用的编译器，即按如下方式修改第38和39行：

```shell
1 Cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
2 -DCMAKE_CXX_COMPILER=icpc\
3 -DMPI_CXX_COMPILER=mpiicpc \
```

保存退出后,运行build_abacus_intel.sh 即可。编译期间可能会有大量ICC相关的Warning,这只是提醒你icpc等旧版 Intel编译器不会在新版 Intel-OneAPI中保持支持，不必理会。通过这一方法，即可基于旧版 Intel-OneAPI编译ABACUS。

有些Intel-OneAPI处于过渡期,如Intel-OneAPI2023.2可能同时具备icpx 和mpiicpc,但不具备 mpicpx。针对这一情况,可以使用另一个编译选项-with-intel-mpi-clas=yes。编辑 toolchain_intel.sh,使install_abacus_toolchain.sh部分呈现如下

```shell
1 ./install_abacus_toolchain.sh \
2 --with-intel=system \
3 --math-mode=mkl \
4 --with-gcc=no \
5 --with-intelmpi=system \
6 --with-cmake=install \
7 --with-scalapack=no \
8 --with-libxc=install \
9 --with-fftw=no \
10 --with-elpa=install \
```

```shell
11 --with-cereal=install \
12 --with-rapidjson=install \
13 --with-libtorch=no \
14 --with-libnpy=no \
15 --with-libri=no \
16 --with-libcomm=no \
17 --with-intel-classic=no \
18 --with-intel-mpi-clas=yes
19 l tee compile.log
```

即可使用 icpx 和 mpicpc 编译 ABACUS 依赖软件。编译 ABACUS 本体时，按照与上述类似的方法在build_abacus_intel.sh 中修改编译器即可。

```shell
1 Cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
2 -DCMAKE_CXX_COMPILER=icpx \
3 -DMPI_CXX_COMPILER=mpiicpc \
```

很好！你已经学会了通过编辑 toolchain_intel.sh 和build_abacus_intel.sh 自定义toolchain以在不同机器条件下编译Intel依赖的ABACUS了，这种自定义同样能用于相关 ABACUS功能插件的引人，如杂化泛函库LibRI,或是DeePKS支持所需 Torch依赖。

## 2.5.3基于Intel Toolchain 引人ABACUS功能插件

在之前的内容中，我们查看并编辑了toolchain_\*.sh和build_abacus_\*.sh 这两个toolchain 核心脚本。通过编辑这两个脚本，我们可以基于 Toolchain 方便快捷地实现不同需求的 ABACUS 编译。其中，toolchain_\*.sh 主要通过调用 install_abacus_toolchain.sh主脚本完成，涉及到几个编译选项，这些编译选项及其作用分为几类：

·-with-PKG=[instal,system,no,[abspath]]，安装特定依赖库时下载并通过下载的软件包安装/从系统环境变量中读取对应依赖库/不使用对应依赖库/[高级]从绝对路径中识别对应依赖库。

·-math-mode=[mkl,openblas,aocl]，指定 Toolchain 会用到的数学库类型，默认为openblas，但如果识别到环境变量 \$MKLROOT，则会切换到 MKL（所以使用GCC Toolchain 时不能加载 OneAPI环境）

·--with-option=[yes,no] 等其他的编译选项，用于对特定依赖库指定不同版本，如 Intel Toolchain 的-with-intel-classic=[yes,no]，或 GNU Toolchain 的-with-openmpi4=[yes,no]

通过README，或运行 install_abacus_toolchain.sh -help，能获取更多相关信息。上面内容中留了一个思考题，其实就只是我在具体编译时使用了-with-cmake=system选项。了解了这些之后，我们就可以很方便地基于Toolchain 编译一些 ABACUS功能插件，下面以 Intel Toolchain为例，简要介绍如何编译支持杂化泛函和 DeePKS的 ABACUS。

## 编译支持杂化泛函计算的ABACUS

在 toolchain_intel.sh中，修改如下两个选项为 install:

1 --with-libri=install \   
2 --with-libcomm=install \

在编辑完其他选项后，运行 toolchain_intel.sh，此时 Toolchain会自动下载安装 LibRI库和LibComm 库，并自动加人到环境变量管理文件 instal/setup 中。运行完毕后，编辑 build_abacus_intel.sh，打开相关代码附近的注释（删除注释符号)，并将相关代码连接起来：

```shell
PREFIX=$ABACUS_DIR
2 ELPA=$INSTALL_DIR/elpa-2025.01.001/cpu
3 # ELPA=$INSTALL_DIR/elpa-2025.01.001/nvidia # for gpu-lcao
CEREAL=$INSTALL_DIR/cereal-master/include/cereal
LIBXC=$INSTALL_DIR/libxc-7.0.0
6 RAPIDJSON=$INSTALL_DIR/rapidjson-master/
# LIBTORCH=$INSTALL_DIR/libtorch-2.1.2/share/cmake/Torch
8 # LIBNPY=$INSTALL_DIR/libnpy-1.0.1/include
9 LIBRI=$INSTALL_DIR/LibRI-O.2.1.0
10 LIBCOMM=$INSTALL_DIR/LibComm-master
11 # DEEPMD=$H0ME/apps/anaconda3/envs/deepmd # v3.0 might have problem
12
13 # Notice: if you are compiling With AMD-CPU or GPU-version ABACUS,then ‘icpc"and‘mpiicpc’ compilers are recol
14 Cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
15 -DCMAKE_CXX_COMPILER=icpx \
16 -DMPI_CXX_COMPILER=mpiicpx\
17 -DMKLROOT=$MKLROOT \
18 -DELPA_DIR=$ELPA \
19 -DCEREAL_INCLUDE_DIR=$CEREAL\
20 -DLibxc_DIR=$LIBXC\
21 -DENABLE_LCAO=ON \
22 -DENABLE_LIBXC=ON \
23 -DUSE_OPENMP=ON\
24 -DUSE_ELPA=ON\
25 -DENABLE_RAPIDJSON=ON \
26 -DRapidJSON_DIR=$RAPIDJSON \
27 -DENABLE_LIBRI=ON \
28 -DLIBRI_DIR=$LIBRI \
29 -DLIBCOMM_DIR=$LIBCOMM\
30# -DENABLE_DEEPKS=1 \
31 # -DTorch_DIR=$LIBTORCH \
32 # -Dlibnpy_INCLUDE_DIR=$LIBNPY \
```

再运行 build_abacus_intel.sh 脚本即可。再次强调：分行符号”后"面不能有任何空格或其他字符，并且各个分行之间不能有注释。作者的经验是，Intel-OneAPI编译的 ABACUS 往往具备更强的OpenMP 并行性能，可以开很多OpenMP线程，通过线程并行加速计算，尤其是跑杂化泛函等需要运行EXX部分的计算时，效率通常来说会比较高。

## 编译支持DeePKS的ABACUS

在 toolchain_intel.sh中，修改如下两个选项为 install:

```shell
1 --with-libtorch=install \
2 --with-libnpy=install \
```

运行 toolchain_intel.sh完毕后，在 build_abacus_intel.sh 中打开相关注释并修改好脚本，比如：

```shell
PREFIX=$ABACUS_DIR
2 ELPA=$INSTALL_DIR/elpa-2025.01.001/cpu
3 # ELPA=$INSTALL_DIR/elpa-2025.01.001/nvidia # for gpu-lcao
4 CEREAL=$INSTALL_DIR/cereal-master/include/cereal
5 LIBXC=$INSTALL_DIR/libxc-7.0.0
6 RAPIDJSON=$INSTALL_DIR/rapidjson-master/
LIBTORCH=$INSTALL_DIR/libtorch-2.1.2/share/cmake/Torch
8 LIBNPY=$INSTALL_DIR/libnpy-1.0.1/include
9 LIBRI=$INSTALL_DIR/LibRI-O.2.1.0
10 LIBCOMM=$INSTALL_DIR/LibComm-master
11 # DEEPMD=$H0ME/apps/anaconda3/envs/deepmd # v3.0 might have problem
12
13 # Notice: if you are compiling with AMD-CPU or GPU-version ABACUS,then `icpc^and‘mpiicpc" compilers are recol
14 CmaKe -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
15 -DCMAKE_CXX_COMPILER=icpx \
16 -DMPI_CXX_COMPILER=mpiicpx \
17 -DMKLROOT=$MKLROOT \
18 -DELPA_DIR=$ELPA \
19 -DCEREAL_INCLUDE_DIR=$CEREAL \
20 -DLibxc_DIR=$LIBXC \
21 -DENABLE_LCAO=ON \
22 -DENABLE_LIBXC=ON \
23 -DUSE_OPENMP=ON \
24 -DUSE_ELPA=ON\
25 -DENABLE_RAPIDJSON=ON \
26 -DRapidJSON_DIR=$RAPIDJSON \
27 -DENABLE_LIBRI=ON \
28 -DLIBRI_DIR=$LIBRI \
29 -DLIBCOMM_DIR=$LIBCOMM \
30 -DENABLE_DEEPKS=1\
31 -DTorch_DIR=$LIBTORCH \
32 -Dlibnpy_INCLUDE_DIR=$LIBNPY \
```

按类似于如上修改的 build_abacus_intel.sh 运行，即可同时编译含LibRI和 DeePKS 支持的 ABACUS。

## 2.6 Toolchain 编译 ABACUS 教程(3)-AMD

## 2.6.1 简介

此前两部 Toochain 相关教程分别介绍如何了使用GCC Toolchain 直接地从头编译 ABACUS，以及如何使用Intel Toolchain利用 Intel OneAPI在Intel-CPU机器上编译具有高效率的 ABACUS，并同时利用Toolchain的可选项编辑方式引入ABACUS 功能插件。这一教程我们将讲解如何在 AMD-CPU 机器上，通过 Toolchain 编译能高效运行的 ABACUS。

## 2.6.2已有方案回顾

## GCC Toolchain

这一方案最为通用，兼容性最好，自然也可以在AMD-CPU机器上完成ABACUS 的编译，编译过程中并没有需要特别调整的选项，按照此前的教程操作即可。

## Intel Toolchain

AMD-CPU机器上其实也是可以使用此前介绍的Intel Toolchain 基于Intel-OneAPI编译ABACUS 的，不过这一做法存在以下问题：

1．编译器只能选用Intel classiccompiler系列，即icicpc/ifort，不能采用Intel-OneAPI 2024.0+之后更新的新版Intel编译器，即icx/icpx/ifx，否则 ELPA 编译是过不去的。

2.在AMD-CPU机器上，这一编译方案的计算效率一般来说较GNU Toolchain偏低（Intel机器则相反），这是源于Intel-OneAPI（主要是 Math Kernel Library）对 Intel-CPU的特化处理。网络上有一些解除这种特化处理带来的限制，让MKL在AMD-CPU上发挥应有性能的办法，但这些方法往往都需要管理员权限和一些额外的技巧，也不推荐普通用户花过多精力去琢磨。

## 2.6.3 AMD Toolchain

## AMD AOCL 和 AOCC 简介

AMD Optimizing CPU Libraries (AOCL)和 AMD Optimizing C/C++ Compiler (AOCC)是 AMD开发的面向高性能计算(HPC)和科学应用的优化工具库，分别对标 Intel的Intel MKL 和IntelCompiler

·AOCL:是一组高度优化的数学库（如BLAS、LAPACK、SCALAPACK,FFTW等），专为AMD EPYC等处理器深度调优，显著提升科学计算、数据分析等任务的运行速度和效率。用户无需修改代码即可通过链接库获得性能提升。AOCL的各个数学库与开源版本的数学库兼容性基本一致。点此进人AOCL 官网。

·AOCC:是基于LLVM/Clang的C/C++ 编译器套件，针对AMD Zen 架构处理器（如 EPYC）进行了深度优化。它利用高级优化技术（如多目标代码生成、循环优化)，帮助开发者编译出在AMD平台上性能出色的应用。点此进入AOCC官网

AOCL 和AOCC 的安装可以参见官网的下载部分，一般来说需要咨询服务器管理员获取其在服务器上的安装位置，并通过 module等方式加载对应的环境变量。本教程中，AOCC 依赖的 AOCL 安装位置为/data/softwares/AMD/5.0.0-aocl-aocc/aocl，其中内容为：

(base) mikoto@amd-cpu:/data/softwares/AMD/5.0.0-aocl-aocc/aocl\\$ ls   
2 amd-blis amd-libs.cfg include_ILP64   
3 amd-compression amd-rng include_LP64   
4 amd-crypto amd-scalapack lib   
5 amd-da amd-securerng lib_ILP64   
6 amd-fftw amd-sparse lib_LP64   
7 amd-libflame amd-utils README   
8 amd-libm aocl-linux-aocc-5.0.0_module set_aocl_interface_symlink.sh   
9 amd-libmem include

AOCC安装位置为/data/softwares/AMD/5.0.0-aocl-aocc/aocc，其中内容为：

1 (base) mikoto@amd-cpu:/data/softwares/AMD/5.0.0-aocl-aocc/aocc\\$ ls   
2 AMDALLOC-LICENSE.TXT FLANG-LICENSE.TXT LICENSERV.TXT   
3 aocc-compiler-5.0.0_module include LICENSE.TXT

4 AOCC-LICENSE.TXT install.sh ompd   
5 A0CC-prerequisites-check.sh lib README   
6 bin lib32 setenv_AOCC.sh   
7FLANG-CHANGELOG.TXT libexec share

通过 AOCL和 AOCC文件夹里面的_module 后缀文件，服务器管理者可以快捷的配置二者的 Modulefile，并便利用户使用。其中 AOCL 的环境变量被命名为AOCLhome，类似于 Intel MKL的MKLROOT。

## 基于AOCC-AOCL的编译方案

1.正确加载 AOCC和 AOCL环境，同时确保系统内没有其他多余环境变量（尤其不能有IntelOneAPI的环境变量)。作者本人采用的方式是：

1 (base) mikoto@amd-cpu:\~\\$ module load aocl-5.0.0-aocc   
2 Loading aocl-5.0.0-aocc   
3 Loading requirement: aocc-5.0.0

作者注：有很多人使用服务器喜欢把各种环境变量相关的 source 和 export命令写在 \$HOME/.bashrc 文件内，甚至有某些服务器运维喜欢把一些环境变量写在更公用的/etc/profile文件内，这都是非常不好的习惯！久而久之，很容易忘记自己到底都加载了哪些环境变量。合理的环境变量管理方式应该是每个需要的环境变量单独做一个.sh 脚本，需要哪个 source 哪个。当然更好的做法是直接用 Modules-env 方式管理。后续“在超算环境编译ABACUS 的建议”教程中有介绍何为Modules。

2．下载ABACUS仓库。

3．进入Toolchain目录

4．运行 toolchain_aocc-aocl.sh 脚本

1 sh ./toolchain_aocc-aocl.sh

如此即可开始基于Toolchain的ABACUS 依赖库编译安装。以Intel Toolchain默认设置为例，会做如下操作：1．检查你的系统GNU编译器版本，其版本不能低于5（2025-02版本Toolchain新功能)，并链接 AMD-AOCC编译器，输出类似于：

WARNING: (./install_abacus_toolchain.sh，line 315) No MPI installation detected (ignore this message in Cr   
2 Checking system GCC version for gcc， intel and amd toolchain   
3 Your System gcc/g++/gfortran version should be consistent   
4 Minimum required version: 5   
5 Your gcc version: 11.4.0   
6 Your g++ version: 11.4.0   
Your gfortran version: 11.4.0   
8 Your GCC version seems to be enough for ABACUS installation.   
Compiling with 128 processes for target native.   
10 Step gcc took 0.00 seconds.   
11 Step intel took 0.00 seconds.   
12 ==: ===== Finding AMD compiler from system paths ==:   
13 path to clang is /data/softwares/AMD/5.0.0-aocl-aocc/aocc/bin/clang   
14 path to clang++ is /data/softwares/AMD/5.0.0-aocl-aocc/aocc/bin/clang++   
15 path to gfortran is /usr/bin/gfortran   
16 Found lib directory /usr/lib/x86_64-linux-gnu

17 CC is /data/softwares/AMD/5.0.0-aocl-aocc/aocc/bin/clang   
18 Check the AMD C compiler path   
19 CXX is /data/softwares/AMD/5.0.0-aocl-aocc/aocc/bin/clang++   
20 Check the AMD C++ compiler path   
21 FC is gfortran   
22 Check the GNU Fortran compiler path   
23 Step amd took 1.00 seconds.

细心的小伙伴可能发现这里链接的 Fortran 编译器是 gfortran 而不是AMD-AOCC 的 flang，原因会在后续说明。

2．下载 OpenBLAS，通过OpenBLAS 内置的 genarch 程序识别系统指令集和架构：

===== Getting proc arch info using OpenBLAS tools ====:   
2 wgethttps://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29 -0 OpenBLAS-0.3.29.tar.gz -no-check   
3 --2025-06-02 23:18:25-- https://codeload.github.com/0penMathLib/0penBLAS/tar.gz/v0.3.29   
4 Resolving codeload.github.com (codeload.github.com)... 20.205.243.165   
5 Connecting to codeload.github.com (codeload.github.com)l20.205.243.165l:443... connected.   
6 HTTP request sent，awaiting response... 200 0K   
Length: unspecified [application/x-gzip]   
8 Saving to: ‘0penBLAS-0.3.29.tar.gz'   
9   
10 OpenBLAS-0.3.29.tar.gz [ <=> ]23.53M 3.69MB/s in8.   
11   
12 2025-06-02 23:18:34 (2.92 MB/s) - ‘0penBLAS-0.3.29.tar.gz’ saved [24671913]   
13   
14 0penBLAS-0.3.29.tar.gz: OK   
15 Checksum of OpenBLAS-0.3.29.tar.gz 0k   
16 OpenBLAS detected LIBCORE = zen   
17 OpenBLAS detected ARCH = x86_64

此处若出现提示：./f_check:100:[: Illegal number:,须知其只是运行 OpenBLAS genarch 程序中可能出现的问题，不影响 Toolchain 编译过程。

3．自动下载和安装CMake编译库以及OpenMPI并行库

==== Installing CMake ======   
2 vget htps://cmake.org/files/v3.31/cmake-3.31.7-linux-x86_64.sh -0 cmake-3.31.7-linux-x86_64.sh --no-chec   
3 cmake-3.31.7-linux-x86_64.sh: OK   
4 Checksum of cmake-3.31.7-linux-x86_64.sh 0k   
5 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/cmake-3.31.7   
6 Step cmake took 26.00 seconds.   
=== Installing OpenMPI ===:   
8 wget https://dowload.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.bz2 -0 openmpi-5.0.7.tar.bz2 -   
9 openmpi-5.0.7.tar.bz2: OK   
10 Checksum of openmpi-5.0.7.tar.bz2 0k   
11 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7   
12 Found directory /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/bin   
13 Found directory /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/lib   
14 Found directory /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/include

15 mpiexec is installed as /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/bin/mpiexec   
16 mpicc is installed as /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/bin/mpicc   
17 mpicxx is installed as /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/bin/mpicxx   
18 mpifort is installed as /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/bin/mpifort   
19 Step openmpi took 541.00 seconds.

## 4．从环境变量中定位并链接 AOCL 中的 BLAS，LAPACK，ScaLAPACK 和 FFTW

5．自动下载并编译安装：数学库 ELPA，泛函库LibXC，其他功能库CEREAL,RapidJSON以及可选功能库：LibRI,LibComm,LibTorch,Libnpy等。后续输出如下：（本案例基于作者本人的AMD 机器完成）

==================== Finding A0CL from system paths ====================   
2 libblis is found in ld search path   
3 libflame is found in ld search path   
4 Found lib directory /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib_LP64   
5 Found lib directory /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib_LP64   
6 Step aocl took 0.00 seconds.   
==================== Installing LIbXC ====================   
8 vget https://gitlab.com/libxc/libxc/-/archive7.0.0/libxc-7.0.0.tar.bz2 -0 libxc-7.0.0.tar.bz2 --no-check   
9 libxc-7.0.0.tar.bz2: 0K   
10 Checksum of libxc-7.0.0.tar.bz2 0k   
11 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/libxc-7.0.0   
12 Step libxc took 20.00 seconds.   
13 ==== Finding FFTW from system paths ====:   
14 libfftw3 is found in ld search path   
15 libfftw3_omp is found in ld search path   
16 libfftw3_mpi is found in ld search path   
17 Found lib directory /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib_LP64   
18 Step fftw took 0.00 seconds.   
19 == ====== Finding ScaLAPACK from system paths ========:   
20 libscalapack is found in ld search path   
21 Found lib directory /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib_LP64   
22 Step scalapack took 1.00 seconds.   
23 ===== Installing ELPA ====== ======   
24 get htps://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.01.001/elpa-2025.01.001.tar.gz -0 e   
25 elpa-2025.01.001.tar.gz: OK   
26 Checksum of elpa-2025.01.001.tar.gz 0k   
27 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/elpa-2025.01.001/cpu   
28 Step elpa took 158.00 seconds.   
29 ==================== Installing CEReAL ====================   
30 ===> Notice: This version of CEREAL is downloaded in GitHub master repository <===   
31 vget htps://codeload.github.com/USCiLab/cereal/tar.gz/master -0 cereal-master.tar.gz -no-check-certific   
32 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/cereal-master   
33 Step cereal took 2.00 seconds.   
34 ========= ========== Installing RAPIDJSON ======== ===   
35 ==> Notice: This version of rapidjson is downloaded in GitHub master repository <===   
36 vget https://codeload.github.com/Tencent/rapidjson/tar.gz/master -0 rapidjson-master.tar.gz --no-check-ce   
37 Installing from scratch into /home/mikoto/abacus-develop/toolchain/install/rapidjson-master   
38 Step rapidjson took 3.00 seconds.

39 Step libtorch took 0.00 seconds.   
40 Step libnpy took 0.00 seconds.   
41 Step libri took 0.00 seconds.   
42 Step libcomm took 0.00 seconds.   
43 ===: ====== usage =====:   
44 Done!   
45 To use the installed tools and libraries and ABACUS version   
46 compiled with it you will first need to execute at the prompt:   
47 source /home/mikoto/abacus-develop/toolchain/install/setup   
48 To build ABACUS by gnu-toolchain， just use:   
49 ./build_abacus_gnu.sh   
50 To build ABACUS by intel-toolchain， just use:   
51 ./build_abacus_intel.sh   
52 To build ABACUS by amd-toolchain in gcc-aocl， just use:   
53 ./build_abacus_gcc-aocl.sh   
54 To build ABACUS by amd-toolchain in aocc-aocl， just use:   
55 ./build_abacus_aocc-aocl.sh   
56 or you can modify the builder scripts to suit your needs.

## 如此即为编译成功。此后，我们通过vim打开build_abacus_aocc-aocl.sh

1 vim ./build_abacus_aocc-aocl.sh

## 查看该文件

```shell
1 #!/bin/bash
2 #SBATCH -J build
3 #SBATCH -N 1
4 #SBATCH -n 16
5 #SBATCH -o instal1.10g
6 #SBATCH -e install.err
7 # JamesMisaka in 2025.03.09
8
9 # Build ABACUS by amd-openmpi toolchain
10
11 # module load openmpi aocc aocl
12
13 ABACUS_DIR=..
14 T00L=$(pwd)
15 INSTALL_DIR=$TOOL/install
16 Source $INSTALL_DIR/setup
17 cd $ABACUS_DIR
18 ABACUS_DIR=$(pwd)
19 #A0CLhome=/opt/aocl-linux-aocc-5.0.0/5.0.0/aocl/ # user should specify this parameter
20
21 BUILD_DIR=build_abacus_aocl
22 rm -rf $BUILD_DIR
23
```

```shell
24 PREFIX=$ABACUS_DIR
25 ELPA=$INSTALL_DIR/elpa-2025.01.001/cpu
26 # ELPA=$INSTALL_DIR/elpa-2025.01.001/nvidia # for gpu-lcao
27 CEREAL=$INSTALL_DIR/cereal-master/include/cereal
28 LIBXC=$INSTALL_DIR/libxc-7.0.0
29 RAPIDJSON=$INSTALL_DIR/rapidjson-master/
30 LAPACK=$AOCLhome/lib
31 SCALAPACK=$AOCLhome/lib
32 FFTW3=$A0CLhome
33 # LIBRI=$INSTALL_DIR/LibRI-0.2.1.0
34 # LIBCOMM=$INSTALL_DIR/LibComm-master
35 # LIBTORCH=$INSTALL_DIR/libtorch-2.1.2/share/cmake/Torch
36 # LIBNPY=$INSTALL_DIR/libnpy-1.0.1/include
37 # DEEPMD=$H0ME/apps/anaconda3/envs/deepmd # v3.0 might have problem
38
39 # if clang++ have problem， switch back to g++
40
41 Cmake -B $BUILD_DIR -DCMAKE_INSTALL_PREFIX=$PREFIX \
42 -DCMAKE_CXX_COMPILER=clang++ \
43 -DMPI_CXX_COMPILER=mpicxx \
44 -DLAPACK_DIR=$LAPACK \
45 -DSCALAPACK_DIR=$SCALAPACK \
46 -DFFTW3_DIR=$FFTW3 \
47 -DELPA_DIR=$ELPA \
48 -DCEREAL_INCLUDE_DIR=$CEREAL \
49 -DLibxc_DIR=$LIBXC \
50 -DENABLE_LCAO=ON \
51 -DENABLE_LIBXC=ON \
52 -DUSE_OPENMP=ON\
53 -DUSE_ELPA=ON \
54 -DENABLE_RAPIDJSON=ON \
55 -DRapidJSON_DIR=$RAPIDJSON \
56# -DENABLE_DEEPKS=1\
57 # -DTorch_DIR=$LIBTORCH \
58 # -Dlibnpy_INCLUDE_DIR=$LIBNPY \
59 # -DENABLE_LIBRI=ON \
60 # -DLIBRI_DIR=$LIBRI \
61 # -DLIBCOMM_DIR=$LIBCOMM \
62 # -DDeePMD_DIR=$DEEPMD \
63 # -DUSE_CUDA=ON \
64 # -DENABLE_CUSOLVERMP=ON\
65# -D CAL_CUS0LVERMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2x.xx/math_libs/1x.x/targets/x86_64-
66
67 # if one want's to include deepmd, your system gcc version should be >= 11.3.0 for glibc requirements
68
69 cmake --build $BUILD_DIR -j ‘nproc"
70 cmake --install $BUILD_DIR 2>/dev/null
71
```

```shell
72 # generate abacus_env.sh
73 cat << EOF > "${TOOL}/abacus_env.sh"
74 #!/bin/bash
75 Source $INSTALL_DIR/setup
76 export PATH="${PREFIX}/bin":\${PATH}
77 eoF
78
79 # generate information
80 cat << EOF
81 ==== ========== usage ========:
82 Done!
83 To use the installed ABACUS version
84 You need to source ${TooL}/abacus_env.sh first !
85
86 EoF
```

我们需要修改 AOCLhome 环境变量为自己的 AOCL 位置，这里需要手动修改的主要原因是 AOCLhome 这个环境变量在AMD AOCL的 modulefile 里面并没有写成全局的形式，说的更具体一点，其 modulefile 里面 AO-CLhome 的定义用的是set指令而非setenv:

1 A0CLhome=/data/softwares/AMD/5.0.0-aocl-aocc/aocl

完成这一修改之后，通过:wq保存并退出，再运行这一脚本

1 sh ./build_abacus_aocc-aocl.sh

即可完成基于AOCC-AOCL的ABACUS 编译，编译过程大致输出如下：

-- The CXx compiler identification is Clang 17.0.6   
-- Detecting CXX compiler ABI info   
-- Detecting CXX compiler ABI info - done   
4 -- Check for working CXx compiler: /data/softwares/AMD/5.0.0-aocl-aocc/aocc/bin/clang++ - skipped   
Detecting CXx compile features   
-- Detecting CXx compile features - done   
RapidJSON found. Headers:   
-- Found Git: /usr/bin/git (found version "2.34.1")   
Found git: attempting to get commit info...   
10 -- Current commit hash: ede143784   
11 Last commit date: Sat May 31 20:20:02 2025 +0800   
12 -- Found Cereal: /home/mikoto/abacus-develop/toolchain/install/cereal-master/include/cereal   
13 Found PkgConfig: /usr/bin/pkg-config (found version "0.29.2")   
14 -- Found ELPA: /home/mikoto/abacus-develop/toolchain/install/elpa-2025.01.001/cpu/lib/libelpa_openmp.so   
15 Performing Test ELPA_VERSION_SATISFIES   
16 -- Performing Test ELPA_VERSION_SATISFIES - Success   
17 -- Found MPI_Cxx: /home/mikoto/abacus-develop/toolchain/install/openmpi-5.0.7/lib/libmpi.so (found version "3.   
18 -- Found MPI: TRUE (found version "3.1")   
19 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD   
20 -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - SucCesS

21 -- Found Threads: TRUE   
22 -- Found OpenMP_CxX: -fopenmp=libomp (found version "5.1")   
23 -- Found OpenMP: TRUE (found version "5.1")   
24 -- Looking for a CUDA compiler   
25 -- Looking for a CUDA compiler - NOTFOUND   
26 -- Found FFTW3: /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib/libfftw3_omp.so   
27 -- Looking for sgemm_   
28 -- Looking for sgemm_ - not found   
29 -- Looking for sgemm_   
30 -- Looking for sgemm_ - found   
31 -- Found BLAS: /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib/libblis.so   
32 -- Looking for cheev_   
33 -- Looking for cheev_ - not found   
34 -- Looking for cheev_   
35 -- Looking for cheev_ - found   
36 -- Found LAPAcK: /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib/libflame.so;/data/softwares/AMD/5.0.0-aocl-aocc/a   
37 -- Found ScaLAPACk: /data/softwares/AMD/5.0.0-aocl-aocc/aocl/lib/libscalapack.so   
38 -- Checking for one of the modules 'libxc'   
39 -- Found Libxc: /home/mikoto/abacus-develop/toolchain/install/libxc-7.0.0/lib/libxc.so   
40 -- Found Libxc: version 7.0.0   
41 -- Configuring done (3.2s)   
42 -- Generating done (0.2s)   
43 -- Build files have been written to: /home/mikoto/abacus-develop/build_abacus_aocl   
44 [ 0%] Building CXX object source/CMakeFiles/driver.dir/driver_run.cpp.0   
45 [0%] Building Cxx object source/module_base/module_container/CMakeFiles/container.dir/ATen/ops/einsum_op.ep.   
46 ：   
47 [100%] Built target lr   
48 [100%] Built target io_basic   
49 [100%] Building Cxx object CMakeFiles/abacus.dir/source/main.cpp.o   
50 [100%] Linking CXX executable abacus   
51 [100%] Built target abacus   
52 -- Install configuration: ""   
53 -- Installing: /home/mikoto/abacus-develop/bin/abacus   
54 = usage   
55 Done!   
56 To use the installed ABACUS version   
57 You need to source /home/mikoto/abacus-develop/toolchain/abacus_env.sh first !

完成编译后，通过toolchain给出的abacus_env.sh环境变量文件，即可加载 ABACUS 相关环境。需注意，从系统内读人的环境变量并不会被写人到 abacus_env.sh 对应的 instal/setup 文件中，因此在实际使用这一abacus时，除了 source abacus_env.sh，我们还需要加载对应的 AOCC-AOCL 环境。

1 # after load A0CC-AOCL env via module-load or source   
2 source abacus_env.sh

如此即可加载 ABACUS 所有依赖库环境和ABACUS 本体，此时可以通过 abacus-version命令确认ABACUS已正确安装和加载：

---

## 第 61–80 页

一 (base) mikoto@amd-cpu:\~/abacus-develop/toolchain\\$ which abacus   
2 /home/mikoto/abacus-develop/bin/abacus   
3 (base) mikoto@amd-cpu:\~/abacus-develop/toolchain\\$ abacus --version   
4 ABACUS version v3.10.0

这样编译就是完成了，此时我们回顾一下toolchain_aocc-aocl.sh 脚本

```shell
(base) mikoto@amd-cpu:~/abacus-develop/toolchain\$ cat toolchain_aocc-aocl.sh
2 #!/bin/bash
3 #SBATCH -J install
4 #SBATCH -N 1
5 #SBATCH -n 16
6 #SBATCH -o compile.log
7 #SBATCH -e compile.err
8
9 # JamesMisaka in 2025-05-05
10 # install abacus dependency by aocc-aocl toolchain
11 # openmpi is recommended to use
12 # libtorch and libnpy are for deepks support， which can be =no
13 # if you want to run EXX calculation, you should set --with-libri=install
14 # gpu-lcao supporting modify: CUDA_PATH and --enable-cuda
15 # export CUDA_PATH=/usr/local/cuda
16
17 ./install_abacus_toolchain.sh \
18 --with-amd=system \
19 --math-mode=aocl \
20 --with-intel=no \
21 --with-gcc=no \
22 --with-openmpi=install \
23 --with-cmake=install \
24 --with-scalapack=system \
25 --with-libxc=install \
26 --with-fftw=system \
27 --with-elpa=install \
28 --with-cereal=install \
29 --with-rapidjson=install \
30 --with-libtorch=no \
31 --with-libnpy=no \
32 --with-libri=no \
33 --with-libcomm=no \
34 --with-4th-openmpi=no \
35 --with-flang=no \
36 | tee compile.log
37 # to use openmpi-version4: set --with-4th-openmpi=yes
```

可以发现几个有趣的点，分别解释如下：

·-with-scalapack,-with-ftw都是 system，并通过这一方案分别链接 AOCL 里面的 ScaLAPACK和FFTW。这点和Intel Toolchain 的-with-scalapack,-with-fftw都是no 不同，因为在那时这些数学库都被打包在 MKL 里

面了。

·-with-openmpi=install:AOCL-AOCC 并不自带 MPI，并推荐采用 OpenMPI作为 MPI库。Tolchain 默认采用OpenMPIv5，如果用户发现 OpenMPI编译存在问题，可以指定-with-4th-openmpi=yes 来下载安装 v4 版本的OpenMPI，这一版本通常兼容性更好，但计算效率会偏低。

·-with-flang=no:若使用 flang 编译器，则 ELPA 的编译是会报错的，虽然可以通过修改 ELPA 源码里面的libtool 文件绕过相关报错，但绕过这些报错之后所编译出来的ABACUS 使用LCAO-genelpa 计算时计算效率相对较差，故 Toolchain 默认方案是将 flang 编译器强行替换为 gfortran。在 Toolchain中，用户可以修改-with-flang=yes 尝试使用flang 编译器编译依赖库并验证这件事情，此时相关源码修改会自动进行。也正因这点，往往 AOCC-AOCL Toolchain 编译得到的 ABACUS 在运行 LCAO-genelpa SCF 计算时，效率不一定比GCC Toolchain 高出很多，且可能存在潜在问题。

## 基于GCC-AOCL的编译方案

如果我们仔细查看 AOCL官网，会发现 AOCL不仅有 AOCC 依赖的包，还有GCC 依赖的包，我们也可以采用 GCC 依赖的AOCL 进行编译，这些编译方案在 Toolchain 里面也提供了支持，具体可见toolchain_gcc-aocl.sh和build_abacus_gcc-aocl.sh 脚本，操作流程与 AOCC-AOCL 方案基本一致。

1．加载 GCC 依赖的 AOCL 环境

2．运行 toolchain_gcc-aocl.sh 脚本

3．安装完成后，编辑 build_abacus_gcc-aocl.sh 脚本中的 AOCLhome 环境变量

4．编辑完成后保存退出，运行 build_abacus_gcc-aocl.sh 完成ABACUS 本体编译

仅有的差别是编译器为GCC 编译器，而 AOCL库是GCC 依赖的。与此同时，基于 Toolchain，如下方案也是可以尝试的：

，可采用 AOCC 依赖的 AOCL 库，通过GCC-AOCL Toolchain 编译 ABACUS 依赖和 ABACUS 本体，这点只需要在加载环境变量的时候加载 AOCC和对应 AOCL库，并在编译时采用GCC-AOCLToolchain的编译脚本等。

. ，可采用AOCC-AOCL Toolchain 编译 ABACUS 依赖库，而用GCC-AOCL编译 ABACUS 本体，这只需要在加载好 AOCC-AOCL 库的环境变量后，先运行 toolchain_aocc-aocl.sh，再编辑并运行 build_abacus_gcc-aocl.sh即可。

## 2.7编译NvidiaGPU版本的ABACUS

## 2.7.1 介绍

本教程旨在介绍ABACUS在支持NVIDIAGPU的服务器上的编译与使用。

## 当前ABACUS支持的GPU并行计算

INPUT 文件中 device 参数需设置为 gpu。

目前 GPU 版本的 ABACUS 仅支持PW 基组的计算，因此 INPUT 文件中 basis_type 参数仅能设置为 pW。

详细请看官方文档：https://abacus.deepmodeling.com/en/latest/advanced/acceleration/cuda.html

LCAO 基组的GPU 计算版本近期会发布。

## NVIDIA GPU & CUDA

NVIDIA GPU(Graphics Processing Unit）是由一种专门设计来处理图形和并行计算任务的电子芯片。自从NVIDIA 推出其首款GPU以来，它们已经从仅仅处理视频游戏图形的设备发展成为能处理各种高性能计算和深度学习任务的强大工具。

CUDA(Compute Unifed Device Architecture),是显卡厂商NVIDIA 推出的运算平台。CUDA是一种由 NVIDIA推出的通用并行计算架构，该架构使NVIDAIGPU能够解决复杂的计算问题。

为了使 ABACUS 的部分功能支持 NVIDIA GPU，ABACUS 开发团队针对NVIDIA GPU写了大量的CUDA代码，使其在NVIDIAGPU上有较高的运行效率。相比于CPU版本的计算任务，有极高的效率提升！

新闻稿：https://mp.weixin.qq.com/s/D8gcQb0bikMdgizLsbvCfQ

## 2.7.2 检查环境

检查当前环境是否有NVIDIAGPU

nvidia-smi命令帮助检测你的服务器是否有NVIDIAGPU设备。

nvcc-version命令帮助你检测你的服务器软件环境中是否包含cuda toolkit

更详细的内容查 看：htps://abacus.deepmodeling.com/en/latest/advanced/acceleration/cuda.html#requiredhardware-software

## 2.7.3编译

如果您已经尝试过使用GCC 编译 ABACUS教程或者 Intel oneAPI编译 ABACUS 教程，那么编译支持在NVIDIAGPU上运行的 ABACUS十分简单。

在保证可以正常编译CPU版本的ABACUS 的环境下，使用：

```shell
1 cmake -B build -DUSE_CUDA=1 # for GCC
2 cd build && make -j'nproc"
3 ＃or
4 CXX=icpx cmake -B build -DUSE_CUDA=1 # for Intel oneAPI
5 cd build && make -j'nproc'h
```

这样编译出来的 abacus 可执行文件就是支持 NVIDIA GPU 运行的 ABACUS。

## 2.8ABACUSLCAO 基组GPU版本使用说明

## 2.8.1 简介

为了提升ABACUS的运行效率，2024年10月发布的ABACUS 3.8版本支持GPU环境下采用LCAO（LinearCombinationof Atomic Orbitals)基组(即数值原子轨道基组)对Kohn-Sham方程进行求解。目前该版本提供GPU支持的模块有广义本征值求解器以及实空间生成哈密顿量的格点积分，这两部分也是LCAO 基组求解 Kohn-Sham方程的热点。在广义本征值求解器方面，目前3.8版本集成了cuSolver，cuSolverMP，ELPA软件，从而可以支持单卡和多卡的GPU加速。在格点积分模块里我们自研了GPU积分加速算法。开发团队未来还会针对LCAO 算的更多模块提供GPU硬件以及其它国产硬件的支持。

## 2.8.2 安装

要使用GPU 版的ABACUS，使用 CMAKE 编译 ABACUS 的时候（ABACUS编译方法详见https://abacus.deepmodeling.com/en/latest/quick_start/easy_install.html)，需要进行如下设置。

·格点积分GPU版支持，以及cusolver求解器支持：需要安装cuda-toolkit，并在编译 ABACUS 时设置-DUSE_CUDA=ON。

·（可选项)cusolvermp 求解器支持：编译 ABACUS 之前，需要确保系统上安装了cusolvermp 相关库，具体安装方法见 https:/docs.nvidia.com/cuda/cusolvermp/。除了设置-DUSE_CUDA=ON 之外，还需要设置-DENABLE_CUSOLVERMP=ON。

·（可选项)GPU版 ELPA 求解器支持：编译 ABACUS 之前，需要确保系统上安装了支持GPU 版本的ELPA,安装方法详见 htps:/github.com/marekandreas/elpa/blob/master/documentation/INSTALL.md，安装过程可以参考https://github.com/deepmodeling/abacus-develop/pull/4969。安装好 ELPA之后，在编译 ABACUS 时，需要设置-DUSE_CUDA=ON,-DUSE_ELPA=ON。

## 2.8.3 Toolchain 安装

Toolchain 也支持安装GPU版 ABACUS 并支持 GPU-LCAO 计算，只需要在 toolchain\*.sh 里打开GPU 相关选项即可自动安装GPU支持的ELPA，以toolchain_intel.sh文件为例：

```shell
1 ./install_abacus_toolchain.sh \
2 --with-intel=system \
3 --math-mode=mkl \
4 --with-gcc=no \
5 --with-intelmpi=system \
6 --with-cmake=install \
7 --with-scalapack=no \
8 --with-libxc=install \
9 --with-fftw=no \
10 --with-elpa=install \
11 --with-cereal=install \
12 --with-rapidjson=install \
13 --with-libtorch=no \
14 --with-libnpy=no \
15 --with-libri=no \
16 --with-libcomm=no \
17 --with-intel-classic=no \
18 --enable-cuda \
19 --gpu-ver=75 \
```

编译完依赖库之后，类似于上面的，在build_abacus_\*.sh 中加入-DUSE_CUDA=ON，同时切换 ELPA 的指向路径（子路径从cpu切换到nvidia)，即可使用 Toolchain 完成 GPU-LCAO 功能的ABACUS 安装。

## 2.8.4使用

## 1.INPUT参数设置

目前 ABACUS 的 LCAO 基组已经GPU 化的模块包括格点积分模块和广义特征值求解模块。这两个模块是否使用GPU 加速可以分开设置。与格点积分和广义特征值求解相关的输入参数包括device以及ks_solver。下面详细介绍一下这两个参数的设置，注意以下介绍基于ABACUS 的LCAO 基组（INPUT文件中的 basis_type 需要设置为lcao)。

·device:device 用于指定 ABACUS 是否使用GPU来加速运算，可以设置为cpu 或者gpu。如果编译的是GPU版的ABACUS（编译时设置了-DUSE_CUDA=ON)，同时机器中至少有一张GPU卡，那么device 的默认值为 gpu,否则为 cpu。

如果device 设置为gpu,那么 ABACUS 将使用GPU来加速计算，对应的，格点积分将使用GPU 加速计算,广义特征值求解器 ks_solver也将默认设置为cusolver,如果您的机器包含多张GPU，也可以将ks_solver 设置为cusolvermp 或者elpa 来调用多卡求解特征值;

如果 device 设置为 cpu，ABACUS 将使用CPU 进行计算，格点积分部分也会使用 CPU 计算，广义特征值求解器ks_solver默认设置为scalapack_gvx,genelpa，或者lapack(默认值详见https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html#ks-solver)

·ks_solver:ks_solver 用于设置广义特征值求解器，目前 LCAO 基组支持的广义特征值求解器有 la-pack,scalapack_gvx,genelpa,cusolver,cusolvermp,elpa。其中 cusolver,cusolvermp 与 elpa 可以调用 GPU 进行广义特征值求解,cusolver 只支持单GPU卡求解，cusolvermp与elpa 支持多GPU卡求解。

在 INPUT文件中设置好device 和ks_solver 参数后，运行 ABACUS，即可调用GPU 进行运算。

## 2.多卡计算

ABACUS 的格点积分模块以及广义特征值求解模块均支持多块GPU卡加速，ABACUS 调用的GPU数量通过 MPI进程数来设置。如果设置n个进程，ABACUS 将会自动调用n张GPU卡来进行计算，若机器中的GPU卡数小于n，ABACUS 则会调用机器中所有GPU卡来进行运算，不推荐进程数设置成大于GPU卡数。需要注意的是，ABACUS 的广义特征值求解部分只有cusolvermp 和 elpa 支持多卡加速，如果ks_solver 设置成 cusolver,无论设置多少个进程，ABACUS 都只会调用一张GPU卡来进行广义特征值求解。

## 2.8.5示例

下面以 tests/performance/P102_si64_lcao的 64个金刚石结构的硅原子体系为例，展示如何使用GPU版 ABA-CUS 进行LCAO 基组下的计算。

环境：GPU/双 3090,CPU/ Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz

ABACUS 版本号：3.8.0,commit:72b1d7ce9

## 1.GPU算例和结果

我们首先调用双GPU卡进行格点积分运算，同时使用cusolver进行广义特征值求解：修改INPUT文件，添加一行“device gpu”,同时将 ks_solver 设置为cusolver,运行命令“OMP_NUM_THREADS=12 mpirun-n 2 abacus”,输出为：

1 Info: Local MPI proc number: 2,0penMP thread number: 12,Total thread number: 24,Local thread limit:   
2   
3 ABACUS v3.8.0   
4   
5 Atomic-orbital Based Ab-initio Computation at UStc   
6   
Website: http://abacus.ustc.edu.cn/   
8 Documentation: https://abacus.deepmodeling.com/   
9 Repository: https://github.com/abacusmodeling/abacus-develp   
10 https://github.com/deepmodeling/abacus-develop   
1 Commit: 72b1d7ce9 (Sat 0ct 5 16:46:53 2024 +0800)   
12   
13 Sat Oct 5 19:04:21 2024   
14 MAKE THE DIR : OUT.autotest/   
15 RUNNING WITH DEVICE : GPU / NVIDIA GeForCe RTX 3090   
16 UNIFORM GRID DIM : 192 \* 192 \* 96   
17 UNIFORM GRID DIM(BIG): 48 \* 48 \* 24

18 DONE(0.652318 SEC） : SETUP UNITCELL   
19 DONE(0.70875 SEC） :SYMMETRY   
20 DONE(0.789308 SEC） : INIT K-POINTS   
21   
22 Self-consistent calculations for electrons   
23   
24 SPIN KPOINTS PROCESSORSNBASE   
251 1 2 832   
26   
27 Use Systematically Improvable Atomic bases   
28   
29 ELEMENT ORBITALS NBASE NATOM xc   
30Si 2s2p1d-8au 13 64   
31   
32 Initial plane wave basis and FFT box   
33   
34 DONE(0.818906 SEC）:INIT PLANEWAVE   
35   
36 SELF-CONSISTENT :   
37   
38gemm_algo_selector::Fastest time: 0.03072 ms   
39 START CHARGE : atomic   
40 DONE(8.17035 SEC) : INIT SCF   
41 \*\*\*\*\*\*   
42 << Start SCF iteration.   
43 ITEr ETOT/eV EDIFF/eV DRHO TIME/s   
44 CU1 -6.83607351e+03 0.00000000e+00 1.7682e-01 13.42   
45 CU2 -6.83164916e+03 4.42435700e+00 1.2487e-01 7.31   
46 cu3 -6.83237991e+03 -7.30748856e-01 7.9826e-03 7.29   
47 Cu4 -6.83238388e+03 -3.97353454e-03 1.2270e-03 7.22   
48 Cu5 -6.83238393e+03 -5.42990673e-05 4.6013e-04 7.26   
49CU6 -6.83238395e+03 -1.43977923e-05 3.8918e-05 7.30   
50CU7 -6.83238395e+03 -3.55773794e-07 1.1467e-05 7.24   
51 cu8 -6.83238395e+03 3.54214631e-09 3.8113e-06 7.23   
52 Cu9 -6.83238395e+03 -2.56457580e-09 6.6704e-08 6.62   
53 >> Leave SCF iteration.   
54 \*\*\*\*\*\*   
55   
56 TOTAL-STRESS (KBAR)   
57   
58 30.8367522660 -0.0000000000 0.0000000000   
59 0.0000000000 75.4069126649 36.2463263984   
60 0.0000000000 36.2463263984 75.4069126649   
61   
62 TOTAL-PRESSURE: 60.550193 KBAR   
63   
64 TIME STATISTICS   
65

<table><tr><td>66</td><td>CLASS_NAME</td><td>NAME</td><td>TIME/s CALLS</td><td></td><td>AVG/sI</td><td> PER/%</td></tr><tr><td>67 68</td><td></td><td>total</td><td>95.57</td><td>11</td><td>8.69</td><td>100.00</td></tr><tr><td>69</td><td> Driver</td><td> reading</td><td>0.13</td><td>1</td><td>0.13</td><td>0.14</td></tr><tr><td>70</td><td> Input_Conv</td><td>Convert</td><td>0.12 1</td><td></td><td>0.12</td><td>0.12</td></tr><tr><td>71</td><td> Driver</td><td>driver_line</td><td>95.43</td><td></td><td>95.43</td><td>99.86</td></tr><tr><td>72</td><td>UnitCell</td><td>check_tau</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>73</td><td>ESolver_KS_LCAO</td><td>before_all_runners</td><td>2.02</td><td>1</td><td>2.02</td><td>2.11</td></tr><tr><td>74</td><td> PW_Basis_Sup</td><td> setuptransform</td><td>0.29</td><td>1</td><td>0.29</td><td>0.30</td></tr><tr><td>75</td><td> PW_Basis_Sup</td><td>distributeg</td><td>0.27</td><td>1</td><td></td><td></td></tr><tr><td>76</td><td> mymath</td><td> heapsort</td><td>0.09</td><td>1097</td><td>0.27</td><td>0.28</td></tr><tr><td>77</td><td> Symmetry</td><td></td><td>0.06</td><td>1</td><td>0.00</td><td>0.09</td></tr><tr><td>78</td><td> PW_Basis_K</td><td> analy_sys</td><td>0.02</td><td>1</td><td>0.06</td><td>0.06</td></tr><tr><td></td><td> PW_Basis_K</td><td>setuptransform</td><td></td><td></td><td>0.02</td><td>0.02</td></tr><tr><td>79</td><td> PW_Basis</td><td>distributeg setup_struc_factor</td><td>0.01</td><td>1</td><td>0.01</td><td>0.01</td></tr><tr><td>80</td><td> NOrbital_Lm</td><td></td><td>1.01</td><td>1</td><td>1.01</td><td>1.06</td></tr><tr><td>81</td><td>Mathzone_Add1</td><td>extra_uniform</td><td>0.04 5</td><td></td><td>0.01</td><td>0.04</td></tr><tr><td>82</td><td> Mathzone_Add1</td><td>SplineD2</td><td>0.00 5</td><td></td><td>0.00</td><td>0.00</td></tr><tr><td>83</td><td>Mathzone_Add1</td><td>Cubic_Spline_Interpolation 0.00</td><td>5</td><td></td><td>0.00</td><td>0.00</td></tr><tr><td>84</td><td> ppcell_vl</td><td>Uni_Deriv_Phi</td><td>0.04 5</td><td></td><td>0.01</td><td>0.04</td></tr><tr><td>85 86</td><td>Ions</td><td>init_vloc</td><td>0.04 1</td><td></td><td>0.04</td><td>0.04</td></tr><tr><td>87</td><td>ESolver_KS_LCA0</td><td>opt_ions</td><td>93.36 1</td><td></td><td>93.36</td><td>97.69</td></tr><tr><td>88</td><td>ESolver_KS_LCA0</td><td> runner</td><td>77 .64 1</td><td></td><td>77 .64</td><td>81.24</td></tr><tr><td>89</td><td>ESolver_KS_LCAO</td><td>before_scf</td><td>6.01 1</td><td></td><td>6.01</td><td>6.29</td></tr><tr><td>90</td><td>ESolver_KS_LCA0</td><td>beforesolver</td><td>0.27 1</td><td></td><td>0.27</td><td>0.28</td></tr><tr><td></td><td> atom_arrange</td><td> set_matrix_grid</td><td>0.19 1</td><td></td><td>0.19</td><td>0.20</td></tr><tr><td>91 92</td><td> Grid_Technique</td><td>search</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>93</td><td>Grid_BigCell</td><td>init</td><td>0.13 1</td><td></td><td>0.13</td><td>0.14</td></tr><tr><td>94</td><td> Record_adj</td><td>grid_expansion_index</td><td>0.01</td><td>2</td><td>0.01</td><td>0.01</td></tr><tr><td></td><td>Grid_Driver</td><td>for_2d</td><td>0.05</td><td>1</td><td>0.05</td><td>0.05</td></tr><tr><td>95 96</td><td> LCAO_domain</td><td> Find_atom</td><td>0.01</td><td>1152</td><td>0.00</td><td>0.01</td></tr><tr><td></td><td>Veff</td><td>grid_prepare</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>97 98</td><td> OverlapNew</td><td>initialize_HR</td><td>0.01</td><td>1</td><td>0.01</td><td>0.01 0.01</td></tr><tr><td>99</td><td>EkineticNew</td><td>initialize_SR</td><td>0.01</td><td>1 1</td><td>0.01</td><td>0.00</td></tr><tr><td>100</td><td>NonlocalNew</td><td>initialize_HR</td><td>0.00</td><td>1</td><td>0.00</td><td>0.03</td></tr><tr><td>101</td><td>Charge</td><td>initialize_HR set_rho_core</td><td>0.03 0.00</td><td>1</td><td>0.03 0.00</td><td>0.00</td></tr><tr><td>102</td><td> Charge</td><td>atomic_rho</td><td>0.98</td><td>2</td><td>0.49</td><td>1.02</td></tr><tr><td>103</td><td> PW_Basis_Sup</td><td>recip2real</td><td>30.81</td><td>74</td><td>0.42</td><td>32.24</td></tr><tr><td>104</td><td> PW_Basis_Sup</td><td> gathers_scatterp</td><td>1.36</td><td>74</td><td>0.02</td><td>1.42</td></tr><tr><td>105</td><td>Potential</td><td>init_pot</td><td>3.73</td><td>1</td><td>3.73</td><td>3.90</td></tr><tr><td>106</td><td>Potential</td><td>update_from_charge</td><td>33.03</td><td>10</td><td>3.30</td><td>34.56</td></tr><tr><td>107</td><td>Potential</td><td>cal_fixed_v</td><td>0.42</td><td>1</td><td>0.42</td><td>0.44</td></tr><tr><td>108</td><td>PotLocal</td><td>cal_fixed_v</td><td>0.41</td><td>1</td><td>0.41</td><td>0.43</td></tr><tr><td>109</td><td> Potential</td><td>cal_v_eff</td><td>32.52</td><td>10</td><td>3.25</td><td>34.03</td></tr><tr><td>110</td><td> H_Hartree_pW</td><td>v_hartree</td><td>5.66</td><td>10</td><td>0.57</td><td>5.92</td></tr><tr><td>111</td><td> PW_Basis_Sup</td><td>real2recip</td><td>8.21</td><td>85</td><td>0.10</td><td>8.59</td></tr><tr><td>112</td><td> PW_Basis_Sup</td><td>gatherp_scatters</td><td>2.45</td><td>85</td><td>0.03</td><td>2.56</td></tr><tr><td>113</td><td>PotXC</td><td>cal_v_eff</td><td>26.7610</td><td></td><td>2.68</td><td>28.00</td></tr><tr><td>114</td><td> XC_Functional</td><td>v_xc</td><td>26.71</td><td>10</td><td>2.67</td><td>27 .95</td></tr><tr><td>115</td><td>Potential</td><td> interpolate_vrs</td><td>0.05</td><td>10</td><td>0.00</td><td>0.05</td></tr><tr><td>116</td><td> Symmetry</td><td> rhog_symmetry</td><td>8.67</td><td>10</td><td>0.87</td><td>9.07</td></tr><tr><td>117</td><td> Symmetry</td><td>group fft grids</td><td>0.70</td><td>10</td><td>0.07</td><td>0.73</td></tr><tr><td>118</td><td> H_Ewald_pw</td><td>compute_ewald</td><td>0.02</td><td>1</td><td>0.02</td><td>0.03</td></tr><tr><td>119</td><td> Charge_Mixing</td><td>init_mixing</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>120</td><td>HSolverLCA0</td><td> solve</td><td>23.39</td><td>9</td><td>2.60</td><td>24.47</td></tr><tr><td>121</td><td> HamiltLCAO</td><td> updateHk</td><td>17.12</td><td>9</td><td>1.90</td><td>17.91</td></tr><tr><td>122</td><td>OperatorLCAO</td><td>init</td><td>15.90</td><td>27</td><td>0.59</td><td>16.63</td></tr><tr><td>123</td><td>Veff</td><td>contributeHR</td><td>11.04</td><td>9</td><td>1.23</td><td>11.55</td></tr><tr><td>124</td><td> Gint_interface</td><td>cal_gint</td><td>16.54</td><td>19</td><td>0.87</td><td>17.30</td></tr><tr><td>125</td><td>Gint_interface</td><td>cal_gint_vlocal</td><td>10.79</td><td>9</td><td>1.20</td><td>11.29</td></tr><tr><td>126</td><td>Gint_k</td><td> transfer_pvpR</td><td>0.25</td><td>9</td><td>0.03</td><td>0.26</td></tr><tr><td>127</td><td> OverlapNew</td><td>calculate_SR</td><td>1.18</td><td>1</td><td>1.18</td><td>1.23</td></tr><tr><td>128</td><td>OverlapNew</td><td>contributeHk</td><td>0.05</td><td>9</td><td>0.01</td><td>0.05</td></tr><tr><td>129</td><td>EkineticNew</td><td>contributeHR</td><td>1.18</td><td>9</td><td>0.13</td><td>1.23</td></tr><tr><td>130</td><td>EkineticNew</td><td>calculate_HR</td><td>1.17</td><td>1</td><td>1.17</td><td>1.23</td></tr><tr><td>131</td><td> NonlocalNew</td><td>contributeHR</td><td>3.59</td><td>9</td><td>0.40</td><td>3.76</td></tr><tr><td>132</td><td> NonlocalNew</td><td>calculate_HR</td><td>3.54</td><td>1</td><td>3.54</td><td>3.71</td></tr><tr><td>133</td><td> OperatorLCAO</td><td>contributeHk</td><td>0.05</td><td>9</td><td>0.01</td><td>0.05</td></tr><tr><td>134</td><td>HSolverLCAO</td><td>hamiltSolvePsiK</td><td>1.27</td><td>9</td><td>0.14</td><td>1.33</td></tr><tr><td>135</td><td> DiagoCusolver</td><td>cusolver</td><td>1.22</td><td>9</td><td>0.14</td><td>1.28</td></tr><tr><td>136</td><td>ElecStateLCA0</td><td> psiToRho</td><td>4.99</td><td>9</td><td>0.55</td><td>5.22</td></tr><tr><td>137</td><td>elecstate</td><td>cal_dm</td><td>0.26</td><td>10</td><td>0.03</td><td>0.27</td></tr><tr><td>138</td><td>psiMulPsiMpi</td><td> pdgemm</td><td>0.25</td><td>10</td><td>0.03</td><td>0.26</td></tr><tr><td>139</td><td> DensityMatrix</td><td>cal_DMR</td><td>0.10</td><td>10</td><td>0.01</td><td>0.10</td></tr><tr><td>140</td><td>Gint</td><td> transfer_DMR</td><td>0.18</td><td>9</td><td>0.02</td><td>0.19</td></tr><tr><td>141</td><td>Gint_interface</td><td>cal_gint_rho</td><td>4.39</td><td>9</td><td>0.49</td><td>4.60</td></tr><tr><td>142</td><td> Charge_Mixing</td><td>get_drho</td><td>0.04</td><td>9</td><td>0.00</td><td>0.04</td></tr><tr><td>143</td><td> Charge</td><td> mix_rho</td><td>4.85</td><td>8</td><td>0.61</td><td>5.07</td></tr><tr><td>144</td><td> Charge</td><td> Broyden_mixing</td><td>0.27</td><td>8</td><td>0.03</td><td>0.29</td></tr><tr><td>145</td><td> ESolver_KS_LCA0</td><td>after_scf</td><td>0.75</td><td>1</td><td>0.75</td><td>0.78</td></tr><tr><td>146</td><td>ModuleIO</td><td> write_rhog</td><td>0.16</td><td>1</td><td>0.16</td><td>0.17</td></tr><tr><td>147</td><td>ESolver_KS_LCAO</td><td>cal_force</td><td>15.72</td><td>1</td><td>15.72</td><td>16.45</td></tr><tr><td>148</td><td> Force_Stress_LCA0 getForceStress</td><td></td><td>15.72</td><td>1</td><td>15.72</td><td>16.45</td></tr><tr><td>149</td><td> Forces</td><td>cal_force_loc</td><td>0.94</td><td>1</td><td>0.94</td><td>0.99</td></tr><tr><td>150</td><td>Forces</td><td>cal_force_ew</td><td>0.77</td><td>1</td><td>0.77</td><td>0.81</td></tr><tr><td>151</td><td>Forces</td><td>cal_force_cc</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>152</td><td>Forces</td><td>cal_force_scc</td><td>1.02</td><td>1</td><td>1.02</td><td>1.06</td></tr><tr><td>153</td><td> Stress_Func</td><td> stress_loc</td><td>0.22</td><td>1</td><td>0.22</td><td>0.23</td></tr><tr><td>154</td><td> Stress_Func</td><td>stress_har</td><td>0.13</td><td>1</td><td>0.13</td><td>0.14</td></tr><tr><td>155</td><td> Stress_Func</td><td> stress_ewa</td><td>0.76</td><td>1</td><td>0.76</td><td>0.79</td></tr><tr><td>156</td><td> Stress_Func</td><td> stress_cc</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>157</td><td> Stress_Func</td><td> stress_gga</td><td>1.74</td><td>1</td><td>1.74</td><td>1.82</td></tr><tr><td>158</td><td>Force_LCAO</td><td>ftable</td><td>10.13</td><td>1</td><td>10.13</td><td>10.60</td></tr><tr><td>159</td><td>Force_LCAO</td><td> allocate</td><td>5.32</td><td>1</td><td>5.32</td><td>5.56</td></tr><tr><td>160</td><td>LCAO_domain</td><td>build_ST_new</td><td>2.55</td><td>2</td><td>1.28</td><td>2.67</td></tr><tr><td>161</td><td>LCAO_domain</td><td>vnl_mu_new</td><td>2.67</td><td>1</td><td>2.67</td><td>2.79</td></tr></table>

162 Force_LCAO cal_fedm 0.06 1 0.06 0.06   
163 Force_LCAO cal_ftvnl_dphi 0.02 1 0.02 0.03   
164 Force_LCAO cal_fvl_dphi 1.36 1 1.36 1.42   
165 Gint_interface cal_gint_force 1.36 1 1.36 1.42   
166 Force_LCAO cal_fvnl_dbeta 3.37 1 3.37 3.52   
167 ESolver_KS_LCA0 cal_stress 0.00 1 0.00 0.00   
168 ESolver_KS_LCA0 after_all_runners 0.00 1 0.00 0.00   
169 ModuleIO write_istate_info 0.00 1 0.00 0.00   
170   
171   
172   
173 START Time : Sat Oct 5 19:04:21 2024   
174 FINISH Time : Sat Oct 5 19:05:56 2024   
175 TOTAL Time :95   
176 SEE INFORMATION IN : OUT.autotest/

## 2.CPU算例和结果

可以看到GPU运行用了95秒，接下来我们调用CPU进行计算，在INPUT文件中设置device为cpu，同时设置ks_solver为 scalapack_gvx，运行命令“OMP_NUM_THREADS=12 mpirun -n2 abacus”,输出:

1 Info: Local MPI proc number: 2,0penMP thread number: 12,Total thread number: 24,Local thread limit:   
2   
3 ABACUS v3.8.0   
4   
5 Atomic-orbital Based Ab-initio Computation at UStc   
6   
Website: http://abacus.ustc.edu.cn/   
8 Documentation: https://abacus.deepmodeling.com/   
9 Repository: https://github.com/abacusmodeling/abacus-develop   
10 htps://github.com/deepmodeling/abacus-develop   
11 Commit: 72b1d7ce9 (Sat 0ct 5 16:46:53 2024 +0800)   
12   
13 Sat 0ct 5 19:06:27 2024   
14 MAKE THE DIR : OUT.autotest/   
15 RUNNING WITH DEVICE : CPU / Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz   
16 UNIFORM GRID DIM : 192 \* 192 \* 96   
17 UNIFORM GRID DIM(BIG): 48 \* 48 \* 24   
18 DONE(0.274561 SEC）: SETUP UNITCELL   
19 DONE(0.330937 SEC）: SYMMETRY   
20 DONE(0.411047 SEC） : INIT K-POINTS   
21   
22 Self-consistent calculations for electrons   
23   
24 SPIN KPOINTS PROCESSORSNBASE   
25 1 1 2 832   
26   
27 Use Systematically Improvable Atomic bases   
28

<table><tr><td>64 65</td><td>CLASS_NAME</td><td>NAME</td><td>TIME/s</td><td>CALLS</td><td>AVG/s</td><td> PER/%</td></tr><tr><td>66</td><td></td><td>total</td><td>224.00 11</td><td></td><td>20.36</td><td>100.00</td></tr><tr><td>67 68</td><td> Driver</td><td> reading</td><td>0.01</td><td>1</td><td>0.01</td><td>0.01</td></tr><tr><td>69</td><td>Input_Conv</td><td>Convert</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>70</td><td>Driver</td><td>driver_line</td><td>223.99 1</td><td></td><td>223.99</td><td>99.99</td></tr><tr><td>71</td><td>UnitCell</td><td>check_tau</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>72</td><td>ESolver_KS_LCA0</td><td> before_all_runners</td><td>1.75</td><td>1</td><td>1.75</td><td>0.78</td></tr><tr><td>73</td><td>PW_Basis_Sup</td><td> setuptransform</td><td>0.03</td><td>1</td><td>0.03</td><td>0.01</td></tr><tr><td>74</td><td> PW_Basis_Sup</td><td>distributeg</td><td>0.02</td><td>1</td><td>0.02</td><td>0.01</td></tr><tr><td>75</td><td> mymath</td><td> heapsort</td><td>0.09</td><td>1097</td><td>0.00</td><td>0.04</td></tr><tr><td>76</td><td> Symmetry</td><td> analy_sys</td><td>0.06</td><td>1</td><td>0.06</td><td>0.03</td></tr></table>

<table><tr><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=1>PW_Basis_K</td><td rowspan=1 colspan=4>setuptransform               0.02  1         0.02</td><td rowspan=1 colspan=2>0.01</td></tr><tr><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1> PW_Basis_K</td><td rowspan=1 colspan=1>distributeg</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=2>1         0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1> PW_Basis</td><td rowspan=1 colspan=1> setup_struc_factor</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=2>1         1.00</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>NOrbital_Lm</td><td rowspan=1 colspan=1>extra_uniform</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=2>5         0.01</td><td rowspan=1 colspan=2>0.02</td></tr><tr><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>Mathzone_Add1</td><td rowspan=1 colspan=1>SplineD2</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>5         0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>Mathzone_Add1</td><td rowspan=1 colspan=1> Cubic_Spline_Interpolation 0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>5         0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1>Mathzone_Add1</td><td rowspan=1 colspan=1>Uni_Deriv_Phi</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=2>5         0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=1> ppcell_vl</td><td rowspan=1 colspan=1>init_vloc</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=2>1         0.04</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>Ions</td><td rowspan=1 colspan=1>opt_ions</td><td rowspan=1 colspan=3>222.19 1        222.19 99.19</td><td rowspan=1 colspan=1>99.19</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1> ESolver_KS_LCAO</td><td rowspan=1 colspan=1> runner</td><td rowspan=1 colspan=3>188.63 1        188.63 84.21</td><td rowspan=1 colspan=2>188.63 84.21</td></tr><tr><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1>ESolver_KS_LCA0</td><td rowspan=1 colspan=1> before_scf</td><td rowspan=1 colspan=1>5.96</td><td rowspan=1 colspan=2>1         5.96</td><td rowspan=1 colspan=1>2.66</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>ESolver_KS_LCA0</td><td rowspan=1 colspan=1>beforesolver</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=2>1         0.20</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>ESolver_KS_LCA0</td><td rowspan=1 colspan=1> set_matrix_grid</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=2>1         0.13</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1> atom_arrange</td><td rowspan=1 colspan=1> search</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>1         0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1> Grid_Technique</td><td rowspan=1 colspan=1>init</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=2>1         0.07</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1> Grid_BigCell</td><td rowspan=1 colspan=1> grid_expansion_index</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>93</td><td rowspan=1 colspan=1> Record_adj</td><td rowspan=1 colspan=1>for_2d</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1> Grid_Driver</td><td rowspan=1 colspan=1> Find_atom</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>1152</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>LCAO_domain</td><td rowspan=1 colspan=1> grid_prepare</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>Veff</td><td rowspan=1 colspan=1>initialize_HR</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1> OverlapNew</td><td rowspan=1 colspan=1>initialize_SR</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1> EkineticNew</td><td rowspan=1 colspan=1>initialize_HR</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1> NonlocalNew</td><td rowspan=1 colspan=1>initialize_HR</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>Charge</td><td rowspan=1 colspan=1> set_rho_core</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1> Charge</td><td rowspan=1 colspan=1>atomic_rho</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>102</td><td rowspan=1 colspan=1> PW_Basis_Sup</td><td rowspan=1 colspan=1> recip2real</td><td rowspan=1 colspan=1>30.88</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>13.78</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1> PW_Basis_Sup</td><td rowspan=1 colspan=1> gathers_scatterp</td><td rowspan=1 colspan=1>1.37</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>Potential</td><td rowspan=1 colspan=1>init_pot</td><td rowspan=1 colspan=1>3.74</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3.74</td><td rowspan=1 colspan=1>1.67</td><td></td></tr><tr><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>Potential</td><td rowspan=1 colspan=1> update_from_charge</td><td rowspan=1 colspan=1>33.04</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>3.30</td><td rowspan=1 colspan=1>14.75</td><td></td></tr><tr><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>Potential</td><td rowspan=1 colspan=1>cal_fixed_v</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=2>1        0.41</td><td rowspan=1 colspan=1>0.19</td><td></td></tr><tr><td rowspan=1 colspan=1>107</td><td rowspan=1 colspan=1>PotLocal</td><td rowspan=1 colspan=1>cal_fixed_v</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.18</td><td></td></tr><tr><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>Potential</td><td rowspan=1 colspan=1>cal_v_eff</td><td rowspan=1 colspan=1>32.59</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>3.26</td><td rowspan=1 colspan=1>14.55</td><td></td></tr><tr><td rowspan=1 colspan=1>109</td><td rowspan=1 colspan=1> H_Hartree_pw</td><td rowspan=1 colspan=1> v_hartree</td><td rowspan=1 colspan=1>5.67</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>2.53</td><td></td></tr><tr><td rowspan=1 colspan=1>110</td><td rowspan=1 colspan=1> PW_Basis_Sup</td><td rowspan=1 colspan=1> real2recip</td><td rowspan=1 colspan=1>8.18</td><td rowspan=1 colspan=2>85       0.10</td><td rowspan=1 colspan=1>3.65</td><td></td></tr><tr><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1> PW_Basis_Sup</td><td rowspan=1 colspan=1> gatherp_scatters</td><td rowspan=1 colspan=1>2.45</td><td rowspan=1 colspan=2>85       0.03</td><td rowspan=1 colspan=1>1.09</td><td></td></tr><tr><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>PotXC</td><td rowspan=1 colspan=1> cal_v_eff</td><td rowspan=1 colspan=1>26.82</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2.68</td><td rowspan=1 colspan=1>11.97</td><td></td></tr><tr><td rowspan=1 colspan=1>113</td><td rowspan=1 colspan=1>XC_Functional</td><td rowspan=1 colspan=1>v_xc</td><td rowspan=1 colspan=1>26.78</td><td rowspan=1 colspan=2>10       2.68</td><td rowspan=1 colspan=1>11.95</td><td></td></tr><tr><td rowspan=1 colspan=1>114</td><td rowspan=1 colspan=1>Potential</td><td rowspan=1 colspan=1> interpolate_vrs</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=2>10       0.00</td><td rowspan=1 colspan=1>0.02</td><td></td></tr><tr><td rowspan=1 colspan=1>115</td><td rowspan=1 colspan=1> Symmetry</td><td rowspan=1 colspan=1> rhog_symmetry</td><td rowspan=1 colspan=1>8.58</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>3.83</td><td></td></tr><tr><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1> Symmetry</td><td rowspan=1 colspan=1>group fft grids</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=2>10       0.07</td><td rowspan=1 colspan=1>0.31</td><td></td></tr><tr><td rowspan=1 colspan=1>117</td><td rowspan=1 colspan=1> H_Ewald_pw</td><td rowspan=1 colspan=1>compute_ewald</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=2>1         0.03</td><td rowspan=1 colspan=2>0.01</td></tr><tr><td rowspan=1 colspan=1>118</td><td rowspan=1 colspan=1> Charge_Mixing</td><td rowspan=1 colspan=1> init_mixing</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.00</td></tr><tr><td rowspan=1 colspan=1>119</td><td rowspan=1 colspan=1> HSolverLCAO</td><td rowspan=1 colspan=1> solve</td><td rowspan=1 colspan=1>134.43 9</td><td rowspan=1 colspan=2>14.94</td><td rowspan=1 colspan=2>60.01</td></tr><tr><td rowspan=1 colspan=1>120</td><td rowspan=1 colspan=1>HamiltLCAO</td><td rowspan=1 colspan=1> updateHk</td><td rowspan=1 colspan=1>71.31 9</td><td rowspan=1 colspan=2>7.92</td><td rowspan=1 colspan=2>31.83</td></tr><tr><td rowspan=1 colspan=1>121</td><td rowspan=1 colspan=1>OperatorLCAO</td><td rowspan=1 colspan=1>init</td><td rowspan=1 colspan=1>70.0827</td><td rowspan=1 colspan=2>2.60</td><td rowspan=1 colspan=2>31.29</td></tr><tr><td rowspan=1 colspan=1>122</td><td rowspan=1 colspan=1>Veff</td><td rowspan=1 colspan=1> contributeHR</td><td rowspan=1 colspan=1>65.229</td><td rowspan=1 colspan=2>7.25</td><td rowspan=1 colspan=2>29.12</td></tr><tr><td rowspan=1 colspan=1>123</td><td rowspan=1 colspan=2> Gint_interface    cal_gint</td><td rowspan=1 colspan=1>135.97 19</td><td rowspan=1 colspan=2>7.16</td><td rowspan=1 colspan=2>60.70</td></tr><tr><td rowspan=1 colspan=1>124</td><td rowspan=1 colspan=5>Gint_interface   cal_gint_vlocal              64.869        7.21</td><td rowspan=1 colspan=2>28.95</td></tr></table>

<table><tr><td>125</td><td>Gint_Tools</td><td>cal_psir_ylm</td><td>22.23</td><td>41472</td><td>0.00</td><td>9.92</td></tr><tr><td>126</td><td>Gint_k</td><td>transfer_pvpR</td><td>0.25</td><td>9</td><td>0.03</td><td>0.11</td></tr><tr><td>127</td><td> OverlapNew</td><td>calculate_SR</td><td>1.19</td><td>1</td><td>1.19</td><td>0.53</td></tr><tr><td>128</td><td> OverlapNew</td><td>contributeHk</td><td>0.04</td><td>9</td><td>0.00</td><td>0.02</td></tr><tr><td>129</td><td>EkineticNew</td><td>contributeHR</td><td>1.19</td><td>9</td><td>0.13</td><td>0.53</td></tr><tr><td>130</td><td> EkineticNew</td><td>calculate_HR</td><td>1.18</td><td>1</td><td>1.18</td><td>0.53</td></tr><tr><td>131</td><td> NonlocalNew</td><td>contributeHR</td><td>3.59</td><td>9</td><td>0.40</td><td>1.60</td></tr><tr><td>132</td><td> NonlocalNew</td><td>calculate_HR</td><td>3.55</td><td>1</td><td>3.55</td><td>1.59</td></tr><tr><td>133</td><td>OperatorLCAO</td><td>contributeHk</td><td>0.05</td><td>9</td><td>0.01</td><td>0.02</td></tr><tr><td>134</td><td> HSolverLCA0</td><td>hamiltSolvePsiK</td><td>10.78</td><td></td><td>1.20</td><td>4.81</td></tr><tr><td>135</td><td> DiagoElpa</td><td>elpa_solve</td><td>10.73</td><td>9</td><td>1.19</td><td>4.79</td></tr><tr><td>136</td><td>ElecStateLCA0</td><td> psiToRho</td><td>52.34</td><td>9</td><td>5.82</td><td>23.37</td></tr><tr><td>137</td><td>elecstate</td><td>cal_dm</td><td>0.16</td><td>10</td><td>0.02</td><td>0.07</td></tr><tr><td>138</td><td> psiMulPsiMpi</td><td> pdgemm</td><td>0.16</td><td>10</td><td>0.02</td><td>0.07</td></tr><tr><td>139</td><td> DensityMatrix</td><td>cal_DMR</td><td>0.09</td><td>10</td><td>0.01</td><td>0.04</td></tr><tr><td>140</td><td>Gint</td><td>transfer_DMR</td><td>0.18</td><td>9</td><td>0.02</td><td>0.08</td></tr><tr><td>141</td><td> Gint_interface</td><td>cal_gint_rho</td><td>51.86</td><td>9</td><td>5.76</td><td>23.15</td></tr><tr><td>142</td><td> Charge_Mixing</td><td> get_drho</td><td>0.04</td><td>9</td><td>0.00</td><td>0.02</td></tr><tr><td>143</td><td> Charge</td><td>mix_rho</td><td>4.86</td><td>8</td><td>0.61</td><td>2.17</td></tr><tr><td>144</td><td> Charge</td><td>Broyden_mixing</td><td>0.28</td><td>8</td><td>0.03</td><td>0.12</td></tr><tr><td>145</td><td>ESolver_KS_LCAO</td><td>after_scf</td><td>0.74</td><td>1</td><td>0.74</td><td>0.33</td></tr><tr><td>146</td><td> ModuleIO</td><td>write_rhog</td><td>0.16</td><td>1</td><td>0.16</td><td>0.07</td></tr><tr><td>147</td><td> ESolver_KS_LCA0</td><td>cal_force</td><td>33.57</td><td>1</td><td>33.57</td><td>14.98</td></tr><tr><td>148</td><td>Force_Stress_LCAO</td><td> getForceStress</td><td>33.56</td><td>1</td><td>33.56</td><td>14.98</td></tr><tr><td>149</td><td>Forces</td><td>cal_force_loc</td><td>0.97</td><td>1</td><td>0.97</td><td>0.44</td></tr><tr><td>150</td><td>Forces</td><td>cal_force_ew</td><td>0.77</td><td>1</td><td>0.77</td><td>0.34</td></tr><tr><td>151</td><td>Forces</td><td>cal_force_cc</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>152</td><td> Forces</td><td>cal_force_scc</td><td>1.05</td><td>1</td><td>1.05</td><td>0.47</td></tr><tr><td>153</td><td> Stress_Func</td><td>stress_loc</td><td>0.23</td><td>1</td><td>0.23</td><td>0.10</td></tr><tr><td>154</td><td> Stress_Func</td><td>stress_har</td><td>0.13</td><td>1</td><td>0.13</td><td>0.06</td></tr><tr><td>155</td><td> Stress_Func</td><td> stress_ewa</td><td>0.75</td><td>1</td><td>0.75</td><td>0.34</td></tr><tr><td>156</td><td> Stress_Func</td><td>stress_cc</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>157</td><td> Stress_Func</td><td>stress_gga</td><td>1.76</td><td>1</td><td>1.76</td><td>0.79</td></tr><tr><td>158</td><td>Force_LCAO</td><td>ftable</td><td>27.88</td><td>1</td><td>27 .88</td><td>12.45</td></tr><tr><td>159</td><td>Force_LCAO</td><td>allocate</td><td>5.29</td><td>1</td><td>5.29</td><td>2.36</td></tr><tr><td>160</td><td> LCAO_domain</td><td>build_ST_new</td><td>2.55</td><td>2</td><td>1.28</td><td>1.14</td></tr><tr><td>161</td><td>LCAO_domain</td><td> vnl_mu_new</td><td>2.65</td><td>1</td><td>2.65</td><td>1.18</td></tr><tr><td>162</td><td>Force_LCAO</td><td>cal_fedm</td><td>0.06</td><td>1</td><td>0.06</td><td>0.03</td></tr><tr><td>163</td><td>Force_LCAO</td><td>cal_ftvnl_dphi</td><td>0.02</td><td>1</td><td>0.02</td><td>0.01</td></tr><tr><td>164</td><td>Force_LCAO</td><td>cal_fvl_dphi</td><td>19.14</td><td>1</td><td>19.14</td><td>8.54</td></tr><tr><td>165</td><td>Gint_interface</td><td>cal_gint_force</td><td>19.14</td><td>1</td><td>19.14</td><td>8.54</td></tr><tr><td>166</td><td>Gint_Tools</td><td>cal_dpsir_ylm</td><td>6.23</td><td>2304</td><td>0.00</td><td>2.78</td></tr><tr><td>167</td><td> Gint_Tools</td><td>cal_dpsirr_ylm</td><td>1.84</td><td>2304</td><td>0.00</td><td>0.82</td></tr><tr><td>168</td><td> Force_LCAO</td><td>cal_fvnl_dbeta</td><td>3.36</td><td>1</td><td>3.36</td><td>1.50</td></tr><tr><td>169</td><td>ESolver_KS_LCA0</td><td>cal_stress</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>170</td><td>ESolver_KS_LCA0</td><td>after_all_runners</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td>171</td><td> ModuleI0</td><td>write_istate_info</td><td>0.00</td><td>1</td><td>0.00</td><td>0.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

173   
174   
175 START Time : Sat 0ct 5 19:06:27 2024   
176 FINISH Time : Sat 0ct 5 19:10:11 2024   
177 TOTAL Time : 224   
178 SEE INFORMATION IN : OUT.autotest/

输出中的“RUNNINGWITHDEVICE”一行标出了此次运算使用的设备，同时运行时间统计里可以看到格点积分以及广义特征值求解的具体耗时，通过这两项耗时可以对比GPU和CPU的计算效率。格点积分耗时对应cal_gint,cal_gint_vlocal,cal_gint_rho.cal_gint_force这几项，广义特征值求解对应hamiltSolvePsiK项（在上面输出中标黄项)。

可以看到GPU运行用了95秒，而同样体系用CPU运行用了224秒，GPU对于该体系有一定的加速效果。

以上就是本教程的内容，希望对学习采用 ABACUS 结合GPU 进行密度泛函理论计算的读者有所帮助，大家也可以在 bohrium 平台按照 ABACUS LCAO 基组GPU 版本使用介绍|Bohrium 跑一遍实际安装使用流程，有问题可以发邮箱给作者（见开头）。

## 2.9在超算环境编译ABACUS的建议

由于不同超算环境提供的软件包版本各不相同，因此我们没办法提供给您确切的安装方法。一般来说，上述文档和 Toolchain 内已提供的安装模式能应对多数的 ABACUS 编译需求，但是某些时候超算环境的特点会使已有方法失效。我们将在本篇文档中介绍一些在超算环境中编译安装 ABACUS 的注意事项和相关知识。如果在具体编译操作过程中遇到了任何问题，请随时联系我们。

## 2.9.1关于 ABACUS

首先，再次简单介绍一下ABACUS。ABACUS（原子算筹）软件同时支持两种基矢量，一种是平面波（Planewave，PW）基组，一种是LCAO (Linear combination of atomic orbitals)或者称 NAO (Numerical atomic orbitals)基组。

用户可以编译仅支持PW基组的ABACUS 或者同时支持两种基组的 ABACUS。同时，针对PW基组的 ABA-CUS，用户还可以选择编译串行版本或并行版本。但因为这个文档的主题是如何在超算环境中编译ABACUS（应该没有人在超算上跑串行的ABACUS吧)，我们接下来只会介绍并行版本的编译方法。如果想要在普通的服务器上安装串行版本的PW基组的 ABACUS，可以参考GCC 编译 ABACUS教程·GitBook 和Intel oneAPI编译ABACUS 教程·GitBook。

## 2.9.2超算平台的简单介绍

## 1.关于 Modules

一般情况下，大多数超算平台是通过EnvironmentModules 来管理基础软件库的。

如何使用 Modules?这里提供一些参考的教程：软件模块使用方法-上海交大超算平台用户手册 Documen-tation

## 2.超算上编译ABACUS的基本路线

在超算平台上编译ABACUS，同样也有两种选择：

一种是基于GCC（the GNU Compiler Collection）编译一种是基于Intel oneAPI或者 Intel Paralel Studio 编译不得不说，使用InteloneAPI将会简单很多！因此，如果你所使用的超算平台正好有InteloneAPI，那么请毫不犹豫的使用它！

## ABACUS依赖的软件库

。 C/C++ 编译器：由于 ABACUS 是由C++语言编写的软件，所以一定需要C++ 编译器。

·Fortran编译器：Fortran 编译器并不直接用于编译ABACUS 软件，而是需要用其编译 ABACUS 所依赖的数学库：BLAS、LAPACK、ScaLAPACK和 ELPA。（这四个软件都是使用Fortran 语言写的，因此需要依赖Fortran编译器）

·MPI库：如果要编译并行版本的 ABACUS，那么MPILibrary是必须的。同时也需要注意，如果你选择自已编译ScaLAPACK和ELPA，那么MPILibrary也是必须的。

·数学库:

·BLAS：BLAS（Basic Linear Algebra Subprograms）是一组基本的线性代数库，用于高效执行常见的基础线性代数运算，如矩阵乘法和向量操作。

·LAPACK：LAPACK（Linear Algebra Package）是一个开源的数值线性代数库，用于解决线性代数问题，包括矩阵分解、方程组求解和特征值计算等。

·FFTW3:FFTW3（Fastest Fourier Transform in the West3）是一个高性能、开源的快速傅里叶变换（FFT）库，用于高效计算各种傅里叶变换和逆变换。

·ScaLAPACK：ScaLAPACK（Scalable LAPACK）是一种并行计算库，构建在LAPACK之上，用于解决大规模线性代数问题，特别适用于分布式和并行计算环境。

·ELPA：ELPA（Eigenvalue SoLvers for Petaflop-Applications）是一个用于高性能计算的开源库，旨在解决大规模高性能计算中的特征值问题，特别是密集矩阵的特征值问题。（用户编译安装abacus 的时候大多在这里出现问题)

·CEREAL：CEREAL是一个C++序列化库，用于将C++ 对象转换为可存储或传输的数据格式，以及将其反序列化回对象。

如果你只编译仅支持 PW 基组的 ABACUS，那么你不需要安装 ScaLAPACK、ELPA 和 CEREAL。他们是LCAO 基组才依赖的软件库。

## 基于Intel oneAPI编译ABACUS

首先在此介绍一下 Intel oneAPI和 Intel Parallel Studio 的区别。用户可能在超算平台上同时看到 Intel oneAPI和Intel Parallel Studio。这是因为Intel Paralle Studio 是oneAPI的前身，或者说 Intel Parallel Studio 是oneAPI的子集。Intel Paralel Studio在2020年已经停止维护，而 Intel oneAPI却是在 2019年11月正式提出。因此我们可以将Intel oneAPI看作是新版的 Parallel Studio 或加强版。

Intel oneAPI（Intel@ oneAPI Toolkits）同时包含 C/C++ 编译器、Fortran 编译器和 Intel MPILibrary。因此如果加载了Intel oneAPI环境，也就不用再次加载其他的MPILibrary。除此之外,Intel oneAPI还包含了Intel? oneAPIMath Kermel Library（简称为 MKL），这个软件库里面包含了大量常用的数学库，包括 ABACUS 依赖的 BLAS、LAPACK、FFTW3 和 ScaLAPACK4个软件库。

因此，如果你只编译支持PW基组的ABACUS，那么Intel oneAPI就已经可以解决所有的依赖问题。

如果你要编译同时支持两种基组的 ABACUS，那么还需要手动编译安装 ELPA和CEREAL，安装完成后，可以参考官方文档中的 ABACUS-Installation-Options 编译 ABACUS。

注意：在使用Intel oneAPI编译ABACUS 的时候,并不意味着软件环境中不需要GCC。refer:Howthe CompilerUses GCC Intel compilers 需要使用GCC 中的头文件和库。因此同时也需要加载 GCC。

## 基于GCC 编译ABACUS

GCC（the GNU Compiler Collection）它同时包含 C/C++ 编译器和Fortran 编译器，但其不包含 MPI Library。因此我们还需要额外的 MPI库（Open MPI或者MPICH)。

由于这是基于GCC 编译 ABACUS，如果用户所使用的超算环境中没有提供 BLAS、LAPACK、FFTW3、ScaLAPACK、ELPA 和CEREAL的modules，那么这些数学库都需要用户一个个手动编译安装。

在安装完成后，可以参考官方文档中的 ABACUS-Installation-Options 编译 ABACUS。

## 2.9.3拓展教程

## 1.ABACUS高级编译选择：

Easy Installation ABACUS documentation

Advanced Installation Options ABACUS documentation

## 2.超算平台使用教程推荐：

上海交大超算平台用户手册

中国科大超算中心用户使用手册

北大超算手册

超算小站

## 2.10ABACUS在曙光DCU集群与AMD 显卡上的编译与使用

## 2.10.1介绍

本教程旨在介绍ABACUS在曙光DCU计算平台上的编译与使用。在AMD显卡上的编译教程详见ABACUSAMD 版本编译教程。

## 1. DCU

DCU (Deep Computing Unit)是一款面向人工智能、科学计算的高性能全功能 GPGPU(General-Purpose com-puting on Graphics Processing Units）加速卡。

中科海光基于DCU硬件提供完整的软件工具链，以DTK(DCU toolkit)为基础软件层为开发者提供运行、编译、调试和性能分析等功能，并提供多种深度优化的计算加速库。DCU加速卡支持ROCm/Hip并行架构。

曙光计算集群采用CPU 和DCU 加速卡（Deep Computing Unit）相结合的异构融合计算体系结构。

## 2.ABACUS的异构并行计算

INPUT 文件中 device 参数需设置为 gpu。

目前 GPU/DCU 版本的 ABACUS 仅支持 PW 基组的计算，因此 INPUT 文件中 basis_type 参数仅能设置为pw。

## 2.10.2准备

## 1.曙光计算平台

用户需要在平台上申请异构计算资源：

https://ac.sugon.com/doc/1.0.6/3oooo/general-handbook/platform/source.html

## 2. E-shell

曙光计算平台采用E-shell来对管理节点进行操作。

可使用网页版或E-Shell Client。

。 不可直接运行任务，而是使用 Slurm 调度系统。

·采用modules工具来管理环境变量与系统依赖项。许多依赖如编译器版本等可以通过modules 来处理。

## 3. slurm

Slurm工作调度工具是面向Linux和Unix及类似内核的免费和开源工作调度程序，可以方便用户进行作业的提交、管理、监测。

·sinfo:查看系统资源。

。 squeue:查看当前作业状态。

·salloc:分配节点的作业提交，用 salloc 申请的节点会在当前终端退出时释放掉。用于程序测试以及中小型任务的提交。

·sbatch:批处理模式的作业提交，需要编写 slurm作业提交脚本。在E-shell的默认目录存有 slurm 脚本模板。在下面的流程中也会介绍提交ABACUS任务的脚本案例。

·srun:交互式提交作业命令，有屏幕输出，但容易受网络波动影响，断网或关闭窗口会导致作业中断。

## 2.10.3流程

## 1.配置超算环境

1 module avail # 列出已有环境  
2 module list # 查看当前已加载环境  
3 module load # 加载环境  
4 module unload # 卸载环境

## 昆山节点

1 module purge   
2 1) compiler/devtoolset/7.3.1   
3 2） compiler/rocm/dtk-22.10   
4 3) compiler/cmake/3.17.2   
5 4) mpi/hpcx/2.6.0/gcc-7.3.1

## 乌镇节点

1 1) compiler/devtoolset/7.3.1   
2 2） compiler/dtk/23.04   
3 3) compiler/cmake/3.23.1   
4 4) mpi/hpcx/gcc-7.3.1

对于使用其他DCU节点(合肥、哈尔滨、西安)的用户,如果module中没有找到类似的环境,欢迎在ABACUS仓库提出issue，我们将尽力协助解决。

## 2.编译ABACUS依赖软件包

目前按照DCU版本已验证的编译方法，有三个数学库需要自行编译。

若曙光平台网络连接不畅，请在软件官网选择合适的软件包，再用曙光平台的E-File 传送至节点。

FFTW:ttps:/ftw.org/pub/fftw/ftw-3.3.10.tr.gzOpenBLAS:htps://ihubcom/xianyi/OpenBLAS/releases/downoad/v03./O 0.3.21.tar.gz ScaLAPACK: htps:/github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.0.targz

## 编译FFTW

```shell
1 tar -zxvf fftw-3.3.10.tar.gz
2
3 mkdir build
4 ./configure --prefix=/work/home/your_username/fftw-3.3.10/build
5 cd ~/fftw-3.3.10/build
6 make
7 make install
```

FFTW需要编译单精度版本和双精度版本：

```shell
1 cd ~/fftw-3.3.10/build
2 ./configure --prefix=/work/home/your_username/fftw-3.3.10/build --enable-float
3 make
4 make install
```

## 编译OpenBLAS

```shell
1 tar -zxvf OpenBLAS-0.3.23.tar.gz
2 cd OpenBLAS-0.3.23
3
4 make USE_OPENMP=1 NO_AVX512=1 FC="gfortran -fPIC" CC="gcc -fPIC" -j8
5 mkdir build
6 make PREFIX=/work/home/your_username/OpenBLAS-0.3.21/build install
```

## 编译ScaLAPACK

```batch
1 cp SLmake.inc.example SLmake.inc
```

复制包中提供的 SLmake.inc.example 作为基准，并对 SLmake.inc 的内容作出一部分修改:

1 FC = mpif90 -fPIC   
2 Cc = mpicc -fPIC   
3 BLASLIB =   
4LAPACKLIB = -L/work/home/your_username/0penBLAS-0.3.21/build/lib -lopenblas

1 make

## 3.编译DCU版本的ABACUS

1 git clone https://gitee.com/deepmodeling/abacus-develop # main 分支   
2 cd abacus-develop   
3 mkdir build   
4 cd build

CMake配置：

指定编译器为clang，关闭OpenMP、LCAO 计算模块，设定三个数学库的位置，设定USE_ROCM=ON。

```shell
CC=clang CXX=clang++ cmake -B build -DUSE_OPENMP=OFF -DENABLE_LCAO=OFF \
2 -DFFTW3_DIR=/work/home/your_username/fftw-3.3.10/build/ \
3 -DLAPACK_DIR=/work/home/your_username/OpenBLAS-0.3.21/build/lib \
4 -DSCALAPACK_DIR=/work/home/your_username/scalapack-2.2.0/ \
5-DUSE_ROCM=ON
```

Make 编译：make部分建议不要并行编译。

1 make

## 4.提交任务

salloc（中小型任务与交互性程序测试）

1 salloc -p [队列名] -N 1 -n 32 --gres=dcu:4  
2  
3 ... load text ...  
4  
5 salloc: Waiting for resource configuration  
6 salloc:Nodes node_num are ready for job # 分配计算节点，可用 ssh 直接连接  
8 ssh node_num  
9  
10..．交互式进行计算任务

## sbatch（大型任务与批量提交）

写一个作业提交脚本，可以参考以下结构：

```shell
1 #!/bin/bash
#SBATCH --job-name=ABACUS_GPU
3 #SBATCH --partition=kshdnormal
#SBATCH --nodes=1
5 #SBATCH --output=output .log
#SBATCH --ntasks-per-node=32
#SBATCH --mail-user=username@email
8 #SBATCH --gres=dcu:4 #dcu 个数
9 #SBATCH --time=01:00:00
10 #SBATCH --error=error.log
1 1
```

12 # 以上的 SBATCH 信息会由 slurm 识别  
13 abacus=/work/home/your_username/abacus-develop/build/abacus_pw  
14 #设置环境  
15 module purge  
16 module load compiler/devtoolset/7.3.1  
17 module load compiler/rocm/dtk-22.10  
18 module load compiler/cmake/3.17.2  
19 module load mpi/hpcx/2.11.0/gcc-7.3.1  
20 ＃ 运行脚本  
21 cd your_task_path  
22 mpirun -np 4 \\$abacus

1 sbatch abacus_dcu.slurm

作业已提交，可在“作业管理”中查看。

## 2.10.4 结语

DCU可以提高ABACUS 计算性能，也充分利用了ROCm并行框架，使得ABACUS 异构计算能应用在更多的平台上。

## 第3章建模

## 3.1ABACUS 结构文件 STRU及其准备方式

## 3.1.1 背景

在使用 ABACUS开展计算之前，我们需要准备好ABACUS 计算用的输人文件，这些输人文件主要包括：

1．STRU文件：ABACUS 的结构文件格式，可类比VASP的POSCAR。

2.INPUT文件：开展 ABACUS 计算的输入参数集合，可类比VASP的INCAR。

3.KPT文件：ABACUS 的K点文件格式，代表了开展第一性原理计算时的K空间采样情况，可类比 VASP的 KPOINTS。

其中，结构文件 STRU的准备是ABACUS 输入文件准备的重要一环。一般来说，用户的结构文件来源主要包括：

1．自行通过 Materials Studio Visualizer 等晶体结构建模可视化软件构建结构模型，并保存为这一建模软件所支持的格式

2．从Materials Project 等晶体结构数据库上下载的结构文件（多为CIF 格式）

3.其他来源，如求解晶体表征谱图得到的CIF文件，从文献中获取的结构文件等。

这些结构文件的格式大多为CIF，POSCAR，XSD等，不会直接是ABACUS的 STRU文件。那么，我们该如何准备ABACUS的结构文件呢？与此同时，在ABACUS计算完成之后，我们有没有将计算结果转化为可被常见的晶体结构建模可视化软件导入并可视化的常见结构文件格式（如CIF）的方法？

## 3.1.2基于ASE-ABACUS 快速准备 STRU文件

ASE（Atomic Simulation Environment）是开源的原子尺度模拟Python工具库，由丹麦技术大学（DTU）等机构开发，广泛应用于计算材料科学和分子模拟领域。该工具库支持多种第一性原理软件，ABACUS 也有 ASE-ABACUS接口，不过这一接口是独立于ASE主仓库以外的分支，需要单独下载安装。安装方式：

```shell
1 # install ase-abacus
2 git clone https://gitlab.com/1041176461/ase-abacus.git
3 cd ase-abacus
4 pip install .
5 # use tsinghua mirror in CN internet if needed
6 # pip install . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

若是直接通过 pip install ase，可以下载到 PyPi上官方的ASE，但这个 ASE 里没有接人 abacus 的部分支持。ASE-ABACUS 是ASE的一个派生分支，专用于ASE的 ABACUS 支持，同时兼容 ASE 本体。安装好了ASE-ABACUS之后就可以使用这个ASE来准备 STRU文件了。有两种可行方式，一种是直接基于ASE构建结构文件并导出为 STRU 格式。比如以下Python 脚本（或 IPython 命令行):

```python
1 from ase.build import bulk
2 from ase.io import write
3
4 structure = bulk("Al", "fcc")
5 write("STRU", structure, format='abacus')
```

如此即可得到 FCC Al的一个最简单的 STRU 文件，它是FCC Al的原胞。

---

## 第 81–100 页

## 3.1.3 STRU 文件的基本结构

通过上述最简单的方式准备的 STRU文件内容如下：

```csv
1 ATOMIC_SPECIES
2 Al 26.9815385 Al_ONCV_PBE-1.0.upf
3
4 NUMERICAL_ORBITAL
5 Al_gga_7au_100Ry_4s4p1d.orb
6
7 LATTICE_CONSTANT
81.8897261258369282
9
10 LATTICE_VECTORS
11 0.0000000000 2.0250000000 2.0250000000
12 2.0250000000 0.0000000000 2.0250000000
132.0250000000 2.0250000000 0.0000000000
14
15 ATOMIC_POSITIONS
16 Direct
17
18 Al
19 0.0000000000
201
21 0.000000000 0.0000000000 0.000000000 1 1 1 mag 0.0
```

可以看到一个基础的 STRU 文件包括如下内容

。 元素类别信息。包括元素符号，摩尔质量，赝势文件名称。

。 原子轨道文件名称。在使用原子轨道基组计算时使用。

·晶格常数单位。ABACUS 的默认长度单位为Bohr，这一晶格常数将所有长度放大1.889726倍，将长度单位Bohr转化为更为人熟知的埃（Angstrom）

。 晶格矢量，为九分量形式。

·原子坐标类别：有多种选项，可以在docs:atomic-positions处查看，常见选项包括分数坐标Direct，笛卡尔坐标Cartesian。其他选项比如 Cartesian_angstrom 选项可以避免对晶格坐标单位LATTICE_CONSTANT 的显式指定，直接以埃为单位，不过这一选项在各个接口软件中支持的不多，也不广为人知。

·每个元素的原子坐标。具体来说：

1 Al＃ 元素类别  
20.0000000000＃ 该元素每个原子的磁矩（低优先级）  
31 # 该元素包含的原子数目  
4 0.0000000000 0.0000000000 0.0000000000 1 1 1 mag 0.0  
5 # 坐标 x,y,z；原子在 x,y,z 方向是否可动(1 代表可动)，原子磁矩(高优先级)

对于不同计算任务，STRU 文件的原子坐标部分会略有不同，比如 nspin=4 时磁矩会有三个分量，跑MD时此 STRU文件会有每个原子的速度信息。该 STRU文件的赝势，轨道文件名称来自于ABACUS 官网上推荐的SG15-v1.0赝势和对应的第二代数值原子轨道，是ASE-ABACUS 输出 STRU 时的默认值。

## 3.1.4 基于ASE-ABACUS从其他结构文件准备 STRU

除了基于ASE构建的结构准备STRU外，更常见的是，用户已经有自己想做计算的输入结构，只是要把它转化为 STRU 格式，这一点ASE-ABACUS 当然也支持，以如下FCC Al的CIF 文件为例：

```csv
1 data_image0
2 _chemical_formula_structural Al
3 _chemical_formula_sum "Al1"
4 _cell_length_a 2.8637824638055176
5 _cell_length_b 2.8637824638055176
6 _cell_length_c 2.8637824638055176
7 _cell_angle_alpha 60.00000000000001
8 _cell_angle_beta 60.00000000000001
9 _cell_angle_gamma 60.00000000000001
10
11 _space_group_name_H-M_alt "P 1"
12 _space_group_IT_number 1
13
14 loop_
15 _space_group_symop_operation_xyz
16 'x，y，z'
17
18 lo0p_
19 _atom_site_type_symbol
20 _atom_site_label
21 _atom_site_symmetry_multiplicity
22 _atom_site_fract_x
23 _atom_site_fract_y
24 _atom_site_fract_z
25 _atom_site_occupancy
26 Al Al1 1.0 0.0 0.0 0.0 1.0000
```

比较简单的Python 操作方式如下：

```python
1 from ase.io import read, write
2 # read existing cif file
3 Al_fcc = read("Al.cif"， format="cif")
4 # write this structure to STRU
5 write("STRU"，Al_fcc，format='abacus')
```

如果想自行指定输出的赝势文件和轨道文件，可以这么写你的Python 脚本：

```python
1 from ase.io import read,write
2 Al_fcc = read("Al.cif", format="cif")
3 # specify pseudopot from APNS-v1.0
4 pp = {"Al": "Al.upf"}
5 # specify orbital from APNS-v1.0
6 basis = {"Al": "Al_gga_8au_100Ry_2s2p1d.orb"}
7 write("STRU"，Al_fcc， format='abacus'， pp=pp, basis=basis)
```

还可以通过设置系统环境变量，让ASE-ABACUS 自动识别你想指定的赝势文件和轨道文件所在路径。以下设置需要写在服务器账户的/.bashrc下

```shell
1 # Set up the environment for ABACUS
2 PP=${HoME}/pseudopotentials # 赝势文件路径
3 ORB=${HOME}/orbitals # 轨道文件路径
export ABACUS_PP_PATH=${PP} # 设置环境变量
5 export ABACUS_ORBITAL_PATH=${ORB} # 设置环境变量
```

## 3.1.5基于ATOMKIT准备 STRU文件

基于ASE-ABACUS准备 STRU文件总还是有它不方便的地方，要么要自行写Python脚本并运行这一脚本，要么要安装 IPython环境才能以命令行交互形式进行STRU文件准备。有没有更便捷的方式呢？

ATOMKIT由VASPKIT开发团队开发，提供了另一种选项：通过命令行交互方式，集成晶体结构操作、建模、格式转化及可视化，并可输出多种常见材料模拟软件结构文件格式。同时，ATOMKIT兼具一定的可视化功能。其最新版可在此处下载：ATOMKIT0.9.0新版发布：新增可视化界面

下载好了之后，直接解压，然后安装即可，如：

```shell
1 # under target directory
2 tar -zxvf atomkit.0.9.0.linux.x64.tar.gz
3 cd atomkit.0.9.0.linux.x64
4 bash setup.sh
5 # reload your system environment
6 source ~/.bashrc
```

如此便可通过atomkit命令启动ATOMKIT:

1 \~> atomkit   
2 \\V/   
3   
4 (l (o)(o） I)   
5 0-----.0000--()--0000.--- 。   
6 - ATOMKIT Standard Edition 0.9.0 (26 Mar. 2024)   
7 Lead Developer: Vei WANG (wangvei@icloud.com)   
8 - Contributors: Zhao-Ke WANG， Gang TANG， Ya-Chao LIU   
9 011111.00001!! 0   
10 （） Dooo. ATOMKIT Made Simple   
11 \（ （）   
12 二 ）/   
13 (_   
14 == Structural Utilities ==:   
15 1) Export Structures 2) Symmetry Analysis   
16 3) Generate K-Mesh & K-Path 4) Edit Structure   
17 5) Advanced Structure Models 6） 2D-Material Kit   
18 8) Visualize Structure 0）Quit   
19 1>>

输入1，即可基于已有结构输出各种格式的结构：

1 1   
2 +! Tip 一   
3 Supplementary parameters from GLOBAL_PARAMETERS.in will be   
4 automatically copied to the exported file, if available.   
5 +-

6 ====== Export Format Options =====:  
101） ABACUS (\*.stru) 102） ABINIT (\*.in)  
8 105） ASE (\*.py) 107） ATAT (\*.in)  
9 109） AtomEye (\*.cfg) 112） CASTEP (\*.cell)  
10 113） CIF Format (\*.cif) 114） CP2K (\*.inp)  
11 115） CPMD (\*.inp) 123） Elk (\*.in)  
12 124） exciting (\*.xml) 126） DFTB+ (\*.hsd)  
13 127） DMol3 (\*.incoor) 129） FHI-aims (\*.in)  
14 132） FLEUR (\*.inp) 134) Gaussian (\*.gjf)  
15 136） LAMMPS (\*.lmp) 138) JDFTx (\*.in)  
16 142） Octopus (\*.in) 144） OpenMX (\*.in)  
17 146）ORCA (\*.in) 149） PDB Format (\*.pdb)  
18 156） Quantum-Espresso (\*.in) 165） RSPt (\*.inp)  
19 168） Siesta (\*.fdf) 175） VASP (\*.vasp)  
20 180） Wannier90 (\*.win) 182） WIEN2K (\*.struct)  
21 185） XCrySDen (\*.xsf) 189） XYZ Format (\*.xyz)  
22  
23 0） Quit  
24 9） Back  
25 1>>

可以看到 ABACUS 的 STRU 格式对应命令101，CIF 格式对应命令113，VSAP的POSCAR 格式对应命令175。输入101即为准备ABACUS 的 STRU文件：

1 101  
2  
3 === ====== Import Format Options ====:  
4 101）ABACUS (STRU) 112） CASTEP (\*.cell)  
5 113） CIF Format (\*.cif) 123） Elk (\*.in)  
6 134) Gaussian (\*.gjf) 142) Materials Studio (\*.xsd)  
7 144） MDL Molfile Format (\*.mol) 149） PDB Format (\*.pdb)  
8 156） Quantum-Espresso (\*.in) 168) Siesta (\*.fdf)  
9 175） VASP (POSCAR，\*.vasp) 185） XCrySDen (\*.xsf)  
10 189） XYZ Format (\*.xyz)  
11 190) Build Crystal from Wyckoff Positions & Space Group Number

13 +- Tip   
14 l Examples of Usage:   
15 - 113 Si.cif: Read structure from Si.cif file with cif format 1   
16 一 175 POSCAR: Read structure from POSCAR file with VASP format 一   
17 - 190 INPUT: Build crystal From Wyckoff positions in INPUT filel   
18 +- +   
19 Input the code-ID listed above and the filename to be read:

然后，按照提示加载我们已有的结构文件。假设当前路径下已有结构文件是Al.cif，如下输人即可：

```diff
1 113 Al.cif
2 -->> (01) Reading Structure from Al.cif File...
3 +-
4 I This utility is in experimental stage & needs to be improved. l
5 +-1 -+
6 +-
7 Species and Wyckoff Sites
8 [A1] 0.000 0.000 0.000
9 +-
10 一 Cell Parameters After Applying Symmetry Operations
11 +1i -+
12 Import File Format: CIF
13 Lattice Constants: 2.864 2.864 2.864
14 Lattice Angles: 60.000 60.000 60.000
15 Total Atoms: 1
16 Number of Atoms: [A1] 1
17 +--
18 -->> (02) Written Al.STRU File!
19 011
20 一 WARNING: Your are using a beta version of ATOMKIT 一
21 一 Please Upgrade to the latest stable version if available 1
22 一 Any Suggestions for Improvement are Welcome and Appreciated
23 01 10
```

可以看到ATOMKIT 自动读取了Al.cif 文件，并将其转化为了ABACUS 的 STRU 格式文件 Al.STRU，这一文件内容为：

```csv
1 ATOMIC_SPECIES
2 Al 26.982 Al_ONCV_PBE-1.0.upf
3
4 NUMERICAL_ORBITAL
5 Al_gga_7au_100Ry_4s4p1d.orb
6
LATTICE_CONSTANT
8 1.889726
9
10 LATTICE_VECTORS
11 2.863782463806 0.000000000000 0.000000000000
12 1.431891231903 2.480108364568 0.000000000000
13 1.431891231903 0.826702788189 2.338268590218
14
15 ATOMIC_POSITIONS
16 Direct
17
18 Al
19 0.000
201
```

21 0.0000000000 0.00000000000 0.000000000000 11 1 ma00.0

读者可自行思考为什么LATTICE_VECTORS 看起来和之前的 STRU文件不同，这里能给出的结论是二者等价。

ATOMKIT有非常丰富的结构操作功能，比如对原子按特定方向排序，按特定方向分层固定原子，交换晶体的晶格向量等，这里不再展开，读者可以自行探索。

## 3.1.6将 STRU文件转化为其他格式

很多时候我们也需要把 STRU文件转化为其他结构格式（如CIF)，以便于在通用的结构可视化软件中导入。使用ASE-ABACUS 进行转化的Python代码如下：

```python
1 from ase.io import read, write
2 # read existing STRU file
3 structure = read("STRU"， format="abacus")
4 # write this structure to CIF
5 write("my_stru.cif", stru, format='cif')
```

而使用ATOMKIT时，操作也和准备 STRU文件类似，只是导出格式选项要输入CIF对应的113，输入文件选项要输入对应的101STRU。同时，ATOMKIT还支持另一种使用方式，用户可以把需要在ATOMKIT中输入的命令写在一个文本文件中，如在example.txt文件内输入如下内容：

1   
2 113   
3 101 STRU

然后通过如下方式

```batch
1 atomkit < example.txt
```

即可一次性完成所有atomkit操作。

## 3.1.7基于abacustest 准备ABACUS 输人文件

除去上述两种方式以外，还有一个工具abacustest也能用来准备ABACUS的 STRU等输入文件。abacustest用于对 ABACUS 进行前后处理，以及高通量任务的计算。目前abacustest 有两种安装方式：通过pip从PyPi源安装，或者是git拉取仓库后源码安装：

1 # install via pip   
pip install abacustest   
3 # install via git+pip   
4 git clone https://github.com/pxlxingliang/abacus-test.git   
cd abacus-test   
6 pip install .

abacustest可以用来高通量地准备ABACUS 的输入任务，高通量地基于Bohrium提交任务，或是高通量地收集整理ABACUS 计算结果。通过abacustest-h，用户可以了解到如何使用这一工具的各个功能。这里仅教学如何准备 STRU 等输入文件。用户需要准备类似于下面的param.json文件：

1{   
2 "prepare" :{   
3 "strus": ["1.vasp", "2.vasp"],   
4 "stru_format": "poscar",   
5 "input_template": "INPUT",   
6 "kpt_template": "KPT",   
7 "pp_dict": {"H": "H.upf","O": "0.upf"},   
8 "orb_dict": {"H": "H.orb"，"O": "0.orb"},   
9 "pp_path": "path/to/pp",   
10 "orb_path": "path/to/orb"   
11 }   
12}

其中,INPUT模板,KPT模板,赝势和轨道的路径等都不是必须指定的,但指定好有利于一次准备好 ABACUS输入。然后，按照如下方式：

```batch
1 abacustest prepare -p param.json -s abacustest
```

即可在 abacustest 文件夹下准备好一系列 ABACUS 输入文件，包含转换好的结构文件 STRU。abacustest 支持对POSCAR 和CIF 等结构格式的转换。

以上内容讲解了ABACUS 结构文件 STRU 的格式，以及通过 ASE-ABACUS，ATOMKIT 和 abacustest 准备ABACUS 的 STRU 文件的方法，同时讲解了如何将 STRU 文件转化为其他结构文件。希望此教程能让用户更快了解ABACUS，并上手ABACUS输入建模和计算！

## 3.2 ABACUS如何选择晶胞朝向获得最佳并行效率？以碳纳米管为例

## 3.2.1 背景

不少用户会问当准备的 ABACUS 算例在XYZ三个方向的大小不一致时，摆放晶胞会不会对结果产生影响？如果会影响，应该怎么选取摆放的方向？相关的 Github Issue 有：htps:/github.com/deepmodeling/abacus-develop/isses/3015。为此，我们这里给了一个包含1000 个碳原子的一维碳纳米管的测试。算例来自：https://github.com/MCresearch/TEAS/tree/main/2022-PRB-CNT/1_DHU/0.01。下面我们分别将真空方向设置为X、Y、Z方向，比较计算结果能量和时间来判断真空方向对计算的影响。

## 3.2.2ABACUS做不同真空方向计算的流程

本文算例的碳纳米管在一个方向的长度明显大于另外两个方向，我们会采用原子弛豫计算(ABACUS 中设置calculation为relax，只让原子移动而固定晶格常数)。改变真空层的方向通过对 STRU和KPT文件的三列进行轮换实现。我们先来看看为ABACUS计算准备的INPUT和KPT输入文件。

## INPUT文件

需要注意的是，上文给出的算例链接中的INPUT文件参数属于 ABACUS 老版本，下面已更新为v3.8.0的参数名称，输入参数关键词可以参考：FullListofINPUTKeywords#

<table><tr><td>3 suffix</td><td>CNT_0.01</td></tr><tr><td>4 calculation</td><td>relax</td></tr><tr><td> nbands</td><td>2100</td></tr><tr><td> symmetry</td><td>0</td></tr><tr><td>7 #Parameters (2.Iteration)</td><td></td></tr><tr><td> ecutwfc</td><td>100</td></tr><tr><td>10 scf_thr</td><td>1e-6</td></tr><tr><td>11 scf_nmax</td><td>400</td></tr><tr><td>12</td><td></td></tr><tr><td>#Parameters (3.Basis)</td><td></td></tr><tr><td>14 basis_type</td><td>lcao</td></tr><tr><td>15 ks_solver</td><td> genelpa</td></tr><tr><td> gamma_only</td><td>1</td></tr><tr><td>17</td><td></td></tr><tr><td>#Parameters (4.Smearing) 18</td><td></td></tr><tr><td> smearing_method</td><td>gaussian</td></tr><tr><td> smearing_sigma</td><td>0.001</td></tr><tr><td>21 #Parameters (5.Mixing)</td><td></td></tr><tr><td>23 mixing_type</td><td></td></tr><tr><td>24 mixing_beta</td><td>pulay</td></tr><tr><td>251 mixing_ndim</td><td>0.1 20</td></tr><tr><td>26</td><td></td></tr><tr><td>27 cal_force</td><td>1</td></tr><tr><td>28 cal_stress</td><td>1</td></tr><tr><td>29 relax_nmax</td><td>1000</td></tr><tr><td>30</td><td>0.04</td></tr><tr><td>force_thr_ev 31 #move_method</td><td>cg</td></tr><tr><td>32 out_stru</td><td>1</td></tr></table>

## KPT文件

1 K_POINTS   
20   
3 Gamma   
4111000

## STRU文件

一共有 1000 个C原子，STRU 文件较长所以不在此处展示，可以在 https://github.com/MCresearch/TEAS/blob/main/2022-PRB-CNT/1_DHU/0.01/STRU 中查看，结构如下图。

<!-- image-->

我们一起看看 STRU文件中的LATTICE_VECTORS，这是一个3×3矩阵，代表我们所计算体系在X，Y，Z三个方向上的晶格矢量大小，还要注意晶格矢量由LATTICE_CONSTANT（晶格常数）进行缩放，此处晶格常数为1Bohr。可以参考：htps://bacus.deepmodeling.com/en/latest/advanced/input_files/stru.html#latice-vectors

```csv
1 LATTICE_CONSTANT
2 1
3 LATTICE_VECTORS
432.1253418058417 0.00000000000000 0.000000000000000
50.00000000000000 34.0150677944206 0.000000000000000
60.000000000000 0.000000000000 202.0428810989945
```

可以观察到此时位于矩阵(3,3)的元素值远大于(1,1)和(2,2)处的值，即此时长边在Z方向。我们要做的就是把长边分别放在不同的方向进行计算，比较效率。

放在X方向时：

1 LATTICE_CONSTANT   
2 1   
3 LATTICE_VECTORS   
4 202.0428810989945 0.00000000000000 0.00000000000000   
50.000000000000000 32.1253418058417 0.00000000000000   
60.000000000000000 0.0000000000000 34.0150677944206

## 放在Y方向时：

```csv
1 LATTICE_CONSTANT
21
3 LATTICE_VECTORS
4 34.01506779442060.000000000000000 0.00000000000000
5 0.00000000000000 202.0428810989945 0.00000000000000
60.00000000000000 0.00000000000000 32.1253418058417
```

原子坐标的轮换也是相同的规则。

由于布里渊区的K点采样选择了Gamma点，故KPT文件轮换后结果不改变，测试中只轮换了STRU 文件中的晶格矢量(LATTICE_VECTORS)和原子坐标(ATOMIC_POSITIONS)两部分来改变晶胞摆放。如果读者在做其他需要变换真空层方向的计算时，千万不要忘记当晶格矢量方向改变时，K点也要改变！

## 3.2.3测试结果

我们选择了64个型号为8426Y+的CPU用64个进程对该一维碳纳米管结构进行结构弛豫。下表给出本次测试的总能量，将碳纳米管摆放在XYZ三个方向下做结构弛豫的步数都是212步。我们通过比较总能量和时间

差异来判断真空层方向对计算结果的影响，可以看到系统总能量并不会受到不同碳纳米管取向的影响，但计算时间会受到显著影响。虽然，但是真空层摆放在X方向时，计算时间明显慢于摆放在Z方向，每一电子步大概慢8s左右。

<table><tr><td rowspan=1 colspan=1>碳纳米管长边朝向</td><td rowspan=1 colspan=1>CPU型号</td><td rowspan=1 colspan=1>CPU个数进程个数</td><td rowspan=1 colspan=1>系统总能量 (eV)</td><td rowspan=1 colspan=1>计算总时间 (s)</td></tr><tr><td rowspan=1 colspan=1>计算总驰豫步数</td><td rowspan=1 colspan=1>运行行一电子步的时间 (s)</td><td rowspan=1 colspan=1>FFT</td><td rowspan=1 colspan=1>UNIFORM GRID DIM(BIG)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>8462Y+</td><td rowspan=1 colspan=1>64核64进程</td><td rowspan=1 colspan=1>-154919.0390601</td><td rowspan=1 colspan=1>64114</td></tr><tr><td rowspan=1 colspan=1>212</td><td rowspan=1 colspan=1>18 18.6</td><td rowspan=1 colspan=1>1296*216*225</td><td rowspan=1 colspan=1>324*54*45</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Y</td><td rowspan=1 colspan=1>8462Y+</td><td rowspan=1 colspan=1>64核64进程</td><td rowspan=1 colspan=1>-154919.0390599</td><td rowspan=1 colspan=1>56846</td></tr><tr><td rowspan=1 colspan=1>212</td><td rowspan=1 colspan=1>15.5 16</td><td rowspan=1 colspan=1>225*1296*216</td><td rowspan=1 colspan=1>45*324*54</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Z</td><td rowspan=1 colspan=1>8462Y+</td><td rowspan=1 colspan=1>64核64进程</td><td rowspan=1 colspan=1>-154919.0390613</td><td rowspan=1 colspan=1>40630</td></tr><tr><td rowspan=1 colspan=1>212</td><td rowspan=1 colspan=1>10 10.5</td><td rowspan=1 colspan=1>216*225*1296</td><td rowspan=1 colspan=1>54*45*432</td><td rowspan=1 colspan=1></td></tr></table>

根据以上测试结果，建议读者们在进行真空层计算时，将碳纳米管的长边摆放在Z方向以获得ABACUS 计算时的最大效率。

## 3.3ABACUS如何选择晶胞朝向获得最佳并行效率？以二维氮化硼为例

## 3.3.1 背景

我们已经测试过了一维材料的长边分别摆放在XYZ三个方向时，对计算效率的影响，并通过测试结果给出了建议，参考 ABACUS 如何选择晶胞朝向获得最佳并行效率？以碳纳米管为例·GitBook。现在我们考虑一个二维六方氮化硼体系，即材料在两个方向的长度明显大于第三个方向。

下面我们分别将短边摆放在X、Y、Z方向，比较计算结果能量和时间来判断短边摆放方式对计算的影响。

## 3.3.2 ABACUS计算的流程

本文算例的氮化硼在一个方向的长度明显小于另外两个方向，我们采用原子弛豫计算(ABACUS 中设置cal-culation为relax，只让原子移动而固定晶格常数)。同样，通过对 STRU和KPT文件的三列进行轮换改变摆放方式。我们先来看看为ABACUS 计算准备的INPUT、STRU和KPT三个输入文件。

## INPUT文件

可以注意到，在本次计算中，我们使用了kspacing 这个参数（Full List of INPUTKeywords—ABACUS doc-umentation），可以根据晶胞大小自动生成 KPT文件。因此不再需要我们手动调换K点的顺序。

1 INPUT_PARAMETERS   
2 suffix bn   
3 pseudo_dir /   
4 orbital_dir /   
5 calculation cell-relax   
6 symmetry 0   
7 basis_type 1cao   
8 ecutwfc 100   
9 scf_thr 1e-07   
10 scf_nmax 100   
11 smearing_method gauss   
12 smearing_sigma 0.002

<table><tr><td>13 mixing_type</td><td>pulay</td></tr><tr><td>14 mixing_beta</td><td>0.3</td></tr><tr><td>15 cal_force</td><td>1</td></tr><tr><td>16 cal_stress</td><td>1</td></tr><tr><td>17 force_thr_ev</td><td>0.01</td></tr><tr><td>18 stress_thr</td><td>0.5</td></tr><tr><td> relax_nmax</td><td>100</td></tr><tr><td> out_stru</td><td>1</td></tr><tr><td>21 1 kspacing</td><td>0.08</td></tr><tr><td></td><td></td></tr></table>

## STRU文件

在本体系中，B和N原子各有96个，h-BN结构图如下。

<!-- image-->

<!-- image-->

以发现，此时短边在晶胞的X方向上。

1 ATOMIC_SPECIES

2 B 10.81

B_ONCV_PBE-1.0.upf

3 N14.007

N_ONCV_PBE-1.0.upf

5 NUMERICAL_ORBITAL

6 B_gga_8au_100Ry_2s2p1d.orb

7 N_gga_8au_100Ry_2s2p1d.orb

9LATTICE_CONSTANT

10 1.8897261258369282

12 LATTICE_VECTORS

13 2.6665300000 0.0000000000 0.0000000000

14 0.0000000000 16.970664000 0.0000000000

15 0.0000000000 0.0000000000 28.020088000

17 ATOMIC_POSITIONS

18 Direct

19B

200.0000000000

21 96

22 0.0000000000 0.0126740000 0.0000000000 1 1 1 mag 0.0

23 0.5000000000 0.018986000 0.0768190000 1 1 1 mag 0.0

24 0.0000000000 0.1439860000 0.0481810000 1 1 1 mag 0.0

<table><tr><td colspan="9">25 0.5000000000 0.1376740000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">26 0.0000000000 0.1439860000 0.2018190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">27 0.5000000000 0.0189860000 0.1731810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">28 0.0000000000 0.2626740000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">29 0.500000000 0.2689860000 0.0768190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">300.0000000000 0.3939860000 0.0481810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">31 0.5000000000 0.3876740000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">320.0000000000 0.3939860000 0.2018190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">33 0.5000000000 0.2689860000 0.1731810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">0.0000000000 0.5126740000 0.0000000000 1 1 1 mag 0.0 34</td></tr><tr><td colspan="9">350.5000000000 0.5189860000 0.0768190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">36 0.0000000000 0.6439860000 0.0481810000 1 1 1 mag 0.0 0.5000000000 0.6376740000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9"></td></tr><tr><td colspan="9">38 0.0000000000 0.6439860000 0.2018190000 1 1 1 mag 0.0 39 0.5000000000 0.5189860000 0.1731810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">400.0000000000 0.7626740000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">41</td></tr><tr><td colspan="9">0.5000000000 0.7689860000 0.0768190000 1 1 1 mag 0.0 42</td></tr><tr><td colspan="9">0.0000000000 0.8939860000 0.0481810000 1 1 1 mag 0.0 43 0.5000000000 0.8876740000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">44 0.0000000000 0.8939860000 0.2018190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">45 0.5000000000 0.7689860000 0.1731810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">46 0.0000000000 0.0126740000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">47 0.5000000000 0.0189860000 0.3268190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">48 0.0000000000 0.1439860000 0.2981810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">49 0.5000000000 0.1376740000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">50 0.0000000000 0.1439860000 0.4518190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">51 0.5000000000 0.0189860000 0.4231810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">52 0.0000000000 0.2626740000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">53 0.5000000000 0.2689860000 0.3268190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">54 0.0000000000 0.3939860000 0.2981810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">55 0.5000000000 0.3876740000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">56 0.0000000000 0.3939860000 0.4518190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">57 0.5000000000 0.2689860000 0.4231810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">58 0.0000000000 0.5126740000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">590.5000000000 0.5189860000 0.3268190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">60 0.0000000000 0.6439860000 0.2981810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">61 0.5000000000 0.6376740000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">62 0.0000000000 0.6439860000 0.4518190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">63 0.5000000000 0.5189860000 0.4231810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">64 0.0000000000 0.7626740000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">65 0.5000000000 0.7689860000 0.3268190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">66 0.0000000000 0.8939860000 0.2981810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">67 0.5000000000 0.8876740000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">68 0.0000000000 0.8939860000 0.4518190000 1 1 1 mag 0.0</td></tr><tr><td colspan="9"></td></tr><tr><td colspan="9">69 0.5000000000 0.7689860000 0.4231810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">70 0.0000000000 0.0126740000 0.5000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">71 0.5000000000 0.0189860000 0.5768190000 1 1 1 mag 0.0 72 0.0000000000 0.1439860000 0.5481810000 1 1 1 mag 0.0</td></tr><tr><td colspan="9" rowspan="5">730.5000000000 0.1376740000 0.6250000000 1 1 1 mag 0.074 0.0000000000 0.1439860000 0.7018190000 1 1 1 mag 0.075 0.5000000000 0.0189860000 0.6731810000 1 1 1 mag 0.076 0.0000000000 0.2626740000 0.5000000000 1 1 1 mag 0.0770.5000000000 0.2689860000 0.5768190000 1 1 1 mag 0.078 0.0000000000 0.3939860000 0.5481810000 1 1 1 mag 0.079 0.5000000000 0.3876740000 0.6250000000 1 1 1 mag 0.0800.0000000000 0.3939860000 0.7018190000 1 1 1 mag 0.0</td></tr><tr><td colspan="1" rowspan="2">1 1 1</td></tr><tr><td colspan="5" rowspan="3">78 0.0000000000 0.3939860000 0.5481810000 1 1 1 mag 0.079 0.5000000000 0.3876740000 0.6250000000 1 1 1 mag 0.0800.0000000000 0.3939860000 0.7018190000 1 1 1 mag 0.0</td><td colspan="1" rowspan="1">111</td></tr><tr><td colspan="2" rowspan="1">0.6250000000</td><td colspan="1" rowspan="1">1 11</td></tr><tr><td colspan="2" rowspan="1">0.7018190000</td><td colspan="1" rowspan="1">111</td></tr><tr><td colspan="5" rowspan="1">81</td><td colspan="2" rowspan="1">0.6731810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="2">820.0000000000 0.5126740000 0.5000000000 1 1 1 mag 0.083 0.5000000000 0.5189860000 0.5768190000 1 1 1 mag 0.0</td><td colspan="2" rowspan="1">0.5000000000</td><td colspan="1" rowspan="1">1 11</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="2" rowspan="1">0.5768190000</td><td colspan="1" rowspan="1">1 11</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="1">84 0.0000000000 0.6439860000 0.5481810000 1 1 1 mag 0.0</td><td colspan="2" rowspan="1">0.5481810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="1">85 0.5000000000 0.6376740000 0.6250000000 1 1 1 mag 0.0</td><td colspan="2" rowspan="1">0.6250000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="1">86 0.0000000000 0.6439860000 0.7018190000 1 1 1 mag 0.0</td><td colspan="2" rowspan="1">0.7018190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">87 0.5000000000</td><td colspan="4" rowspan="1">0.5189860000</td><td colspan="2" rowspan="1">0.6731810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">88 0.0000000000</td><td colspan="4" rowspan="1">0.7626740000</td><td colspan="2" rowspan="1">0.5000000000</td><td colspan="1" rowspan="1">1 11</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">89 0.5000000000</td><td colspan="4" rowspan="1">0.7689860000</td><td colspan="2" rowspan="1">0.5768190000</td><td colspan="1" rowspan="1">1 11</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">900.0000000000</td><td colspan="4" rowspan="1">0.8939860000</td><td colspan="2" rowspan="1">0.5481810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">91 0.5000000000</td><td colspan="4" rowspan="1">0.8876740000</td><td colspan="2" rowspan="1">0.6250000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">920.0000000000</td><td colspan="4" rowspan="1">0.8939860000</td><td colspan="2" rowspan="1">0.7018190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">93 0.5000000000</td><td colspan="4" rowspan="1">0.7689860000</td><td colspan="2" rowspan="1">0.6731810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">94 0.0000000000</td><td colspan="4" rowspan="1">0.0126740000</td><td colspan="2" rowspan="1">0.7500000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">95 0.5000000000</td><td colspan="4" rowspan="1">0.0189860000</td><td colspan="2" rowspan="1">0.8268190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">96 0.0000000000</td><td colspan="4" rowspan="1">0.1439860000</td><td colspan="2" rowspan="1">0.7981810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">97</td><td colspan="4" rowspan="1">0.1376740000</td><td colspan="2" rowspan="1">0.8750000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">98 0.0000000000</td><td colspan="4" rowspan="1">0.1439860000</td><td colspan="2" rowspan="1">0.9518190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">99 0.5000000000</td><td colspan="4" rowspan="1">0.0189860000</td><td colspan="2" rowspan="1">0.9231810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">1000.0000000000</td><td colspan="4" rowspan="1">0.2626740000</td><td colspan="2" rowspan="1">0.7500000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">101</td><td colspan="4" rowspan="1">0.2689860000</td><td colspan="2" rowspan="1">0.8268190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">102</td><td colspan="4" rowspan="1">0.3939860000</td><td colspan="2" rowspan="1">0.7981810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">103 0.5000000000</td><td colspan="4" rowspan="4">0.38767400</td><td colspan="2" rowspan="1">0.8750000000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="3">104</td><td colspan="2" rowspan="3">986000</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2" rowspan="2">000</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2" rowspan="1">0.9518190000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">105</td><td colspan="4" rowspan="1">0.2689860000</td><td colspan="2" rowspan="1">0.9231810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="3">1060.0000000000 0.5126740000 0.750000000 1 1 1 mag 0.00.5000000000 0.5189860000 0.8268190000 1 1 1 mag 0.00.0000000000 0.6439860000 0.7981810000 1 1 1 mag 0.0</td><td colspan="2" rowspan="2">0.750000</td><td colspan="1" rowspan="1">1 11</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="1"> mag 0.0</td></tr><tr><td colspan="5" rowspan="2">0.0000000000 0.6439860000 0.7981810000 1 1 1 mag 0.00.5000000000 0.6376740000 0.8750000000 1 1 1 mag 0.0</td><td colspan="2" rowspan="1">0.7981810000</td><td colspan="1" rowspan="1">111</td><td colspan="1" rowspan="2">111 mg0.0</td></tr><tr><td colspan="2" rowspan="1">0.8750000000</td><td></td></tr><tr><td colspan="9" rowspan="1">1100.0000000000 0.6439860000 0.9518190000 1 1 1 mag 0.00.5000000000 0.5189860000 0.9231810000 1 1 1 mag 0.0120.0000000000 0.7626740000 0.7500000000 1 1 1 mag 0.00.5000000000 0.7689860000 0.8268190000 1 1 1 mag 0.01140.0000000000 0.8939860000 0.7981810000 1 1 1 mag 0.0115 0.5000000000 0.8876740000 0.8750000000 1 1 1 mag 0.01160.0000000000 0.8939860000 0.9518190000 1 1 1 mag 0.01170.5000000000 0.7689860000 0.9231810000 1 1 1 mag 0.0118119 N1200.0000000000</td></tr><tr><td colspan="9">121 96</td></tr><tr><td colspan="9">122 0.0000000000 0.0518080000 0.0478800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">123 0.5000000000 0.1768080000 0.0771200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">124 0.0000000000 0.1807370000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">125 0.5000000000 0.1768080000 0.1728800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">126 0.0000000000 0.0518080000 0.2021200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">127 0.5000000000 0.0557370000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">128 0.0000000000 0.3018080000 0.0478800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">129 0.5000000000 0.4268080000 0.0771200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">130 0.0000000000 0.4307370000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">131 0.5000000000 0.4268080000 0.1728800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">132 0.0000000000 0.3018080000 0.2021200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">133 0.5000000000 0.3057370000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">134 0.0000000000 0.5518080000 0.0478800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">135 0.5000000000 0.6768080000 0.0771200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">136 0.0000000000 0.6807370000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">137 0.500000000 0.6768080000 0.172880000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">138 0.0000000000 0.5518080000 0.2021200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">139 0.5000000000 0.5557370000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">140 0.0000000000 0.8018080000 0.0478800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">141 0.5000000000 0.9268080000 0.0771200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">142 0.0000000000 0.9307370000 0.0000000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">143 0.5000000000 0.9268080000 0.172880000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">144 0.0000000000 0.8018080000 0.2021200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">145 0.5000000000 0.8057370000 0.1250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">146 0.0000000000 0.0518080000 0.2978800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">147 0.5000000000 0.1768080000 0.3271200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">148 0.0000000000 0.1807370000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">149 0.5000000000 0.1768080000 0.4228800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">150 0.0000000000 0.0518080000 0.4521200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">151 0.5000000000 0.0557370000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">152 0.0000000000 0.3018080000 0.2978800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">153 0.5000000000 0.4268080000 0.3271200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">154 0.000000000 0.4307370000 0.25000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">155 0.5000000000 0.4268080000 0.4228800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">156 0.0000000000 0.3018080000 0.4521200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">0.500000000 0.3057370000 0.3750000000 1 1 1 mag 0.0 157</td></tr><tr><td colspan="9">158 0.0000000000 0.5518080000 0.2978800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">159 0.5000000000 0.6768080000 0.3271200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">160 0.0000000000 0.6807370000 0.2500000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">161 0.5000000000 0.6768080000 0.4228800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">162 0.0000000000 0.5518080000 0.4521200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">163 0.5000000000 0.5557370000 0.3750000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">164 0.0000000000 0.8018080000 0.2978800000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">165 0.5000000000 0.9268080000 0.3271200000 1 1 1 mag 0.0</td></tr><tr><td colspan="9"></td></tr><tr><td colspan="9">166 0.0000000000 0.9307370000 0.250000000 1 1 1 mag 0.0</td></tr><tr><td colspan="9">167 0.5000000000 0.9268080000 0.4228800000 1 1 1 mag 0.0 168 0.0000000000 0.8018080000 0.4521200000 1 1 1 mag 0.0</td></tr></table>

<table><tr><td>169</td><td>0.5000000000 0.8057370000 0.3750000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>170 0.0000000000 0.0518080000 0.5478800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>171 0.5000000000 0.1768080000 0.5771200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>172 0.0000000000 0.1807370000 0.5000000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>173 0.5000000000 0.1768080000 0.6728800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>174 0.0000000000 0.0518080000 0.7021200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>175 0.5000000000 0.0557370000 0.6250000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>176 0.0000000000 0.3018080000 0.5478800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>177 0.5000000000 0.4268080000 0.5771200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>178 0.0000000000 0.4307370000 0.5000000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>179 0.5000000000 0.4268080000 0.6728800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>180 0.0000000000 0.3018080000 0.7021200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>181 0.5000000000 0.3057370000 0.6250000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>182 0.0000000000 0.5518080000 0.5478800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>183 0.5000000000 0.6768080000 0.5771200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>184 0.0000000000 0.6807370000 0.5000000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>185 0.5000000000 0.6768080000 0.6728800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>186 0.0000000000 0.5518080000 0.7021200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>187 0.5000000000 0.5557370000 0.6250000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>188 0.0000000000 0.8018080000 0.5478800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>189 0.5000000000 0.9268080000 0.5771200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>190 0.0000000000 0.9307370000 0.5000000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>191 0.5000000000 0.9268080000 0.672880000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>192 0.0000000000 0.8018080000 0.7021200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>193 0.5000000000 0.8057370000 0.6250000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>194 0.0000000000 0.0518080000 0.797880000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>195 0.5000000000 0.1768080000 0.8271200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>196 0.0000000000 0.1807370000 0.7500000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>197 0.5000000000 0.1768080000 0.9228800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>198 0.0000000000 0.0518080000 0.9521200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>199 0.5000000000 0.0557370000 0.8750000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td>201</td><td>200 0.0000000000 0.3018080000 0.7978800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>0.5000000000 0.4268080000 0.8271200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>202 0.0000000000 0.4307370000 0.7500000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>203 0.5000000000 0.4268080000 0.9228800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>204 0.0000000000 0.3018080000 0.9521200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>205 0.5000000000 0.3057370000 0.8750000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td>207</td><td>206 0.0000000000 0.5518080000 0.7978800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>0.5000000000 0.6768080000 0.8271200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>208 0.0000000000 0.6807370000 0.7500000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>209 0.5000000000 0.6768080000 0.9228800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>210 0.0000000000 0.5518080000 0.9521200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td>211</td><td>0.5000000000 0.5557370000 0.8750000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>212 0.0000000000 0.8018080000 0.7978800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>213 0.5000000000 0.9268080000 0.8271200000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>214 0.0000000000 0.9307370000 0.7500000000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>215 0.5000000000 0.9268080000 0.9228800000 1 1 1 mag 0.0</td><td></td><td></td></tr><tr><td></td><td>216 0.0000000000 0.8018080000 0.9521200000 1 1 1 mag 0.0</td><td></td><td></td></tr></table>

```html
217 0.5000000000 0.8057370000 0.8750000000 1 1 1 mag 0.0
```

改变摆放方式时，要同时轮换LATTICE_VECTORS 和 ATOMIC_POSITIONS 的顺序。

## KPT文件

```csv
1 K_POINTS
3 Gamma
41632000
```

如前所述，该KPT文件是由INPUT文件里的kspacing 参数所生成，因此不需要我们手动轮换，每次计算都会自动根据晶胞大小产生合适的K点。

## 3.3.3测试结果

计算采用ABAUCSv3.8.0版本，选择64个型号为8426Y+的CPU用64个进程对该二维氮化硼体系进行结构弛豫。下表给出本次测试的结果，将氮化硼的短边分别摆放在XYZ三个方向完成结构弛豫的步数都是9步。我们通过比较总能量和时间差异来判断摆放方式对计算结果的影响，可以看到系统总能量并不会受到短边取向的影响，但计算时间会受到显著影响。短边摆放在Z方向时，即两长边分别放在X、Y方向，计算时间明显慢于长边摆放在Y、Z和Z、X方向，每一电子步大概慢6s左右。（注意：我们进行一维材料计算时，也建议将长边摆放在Z方向以提高计算效率)

<table><tr><td rowspan=1 colspan=1>h-BN 的两个长边方向</td><td rowspan=1 colspan=1>系统总能量(eV)</td><td rowspan=1 colspan=1>计算总时间 (s)</td><td rowspan=1 colspan=1>计算总驰豫步数</td><td rowspan=1 colspan=1>运行一电子步的时间 (s)</td></tr><tr><td rowspan=1 colspan=1>FFT</td><td rowspan=1 colspan=1>UNIFORMGRID DIM(BIG)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>XY</td><td rowspan=1 colspan=1>-33578.4871085887</td><td rowspan=1 colspan=1>2140</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>24.8 26.8</td></tr><tr><td rowspan=1 colspan=1>216*360*36</td><td rowspan=1 colspan=1>54*72*18</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>YZ</td><td rowspan=1 colspan=1>-33578.4871085889</td><td rowspan=1 colspan=1>1609</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>18.7 20.0</td></tr><tr><td rowspan=1 colspan=1>36*216*360</td><td rowspan=1 colspan=1>9*54*120</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ZX</td><td rowspan=1 colspan=1>-33578.4871085888</td><td rowspan=1 colspan=1>1629</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>18.9 20.9</td></tr><tr><td rowspan=1 colspan=1>360*36*216</td><td rowspan=1 colspan=1>72*9*54</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

根据以上测试结果，建议读者们在进行二维体系计算时，将短边摆放在X或Y方向以获得 ABACUS 计算时的最大效率。

## 3.4 ABACUS如何选择晶胞朝向获得最佳并行效率？以铜表面一氧化碳吸附为例

## 3.4.1 背景

不少用户会问当准备的ABACUS 算例在XYZ三个方向的大小不一致时，摆放晶胞会不会对结果产生影响？如果会影响，应该怎么选取摆放的方向？相关的Github Issue有：abacus/issues/3015。此前已有针对一维材料和（不含真空层的）二维材料其不同晶胞朝向的计算效率测试，并通过测试结果给出了最佳计算效率方案，但这两种情况并没有囊括所有的用户需求，其中一个非常重要的需求是表面体系和表面反应体系。这类体系虽然和二维材料体系一样是二维体系，但其在垂直于表面的方向的晶格长边（一般为Z轴方向）是具有不含有任何原子的真空层的，这一真空层厚度一般在10-15 Angstrom，从而使得真空层方向显著长于另外两个表面延展方向。

下面我们分别将真空方向设置为X、Y、Z方向，比较计算结果能量和时间来判断真空方向对计算的影响。

值得强调的是,后续晶胞构建,ABACUS 计算和结果提取全部由ABACUS 智能体完成,这一智能体以GaliLeo科学智能体平台为智能体平台，以ABACUS-agent-tools中的智能体工具实现为基础构建。GaliLeo 科学智能体平台聚焦“自然语言+科研计算”，用户可在平台网页通过自然语言向智能体ADAM提出需求，如“计算蛋白-小分子结合自由能”、“计算分子在表面的吸附能”等，ADAM将自动解析→查询 ADAM-KB 知识库→调用HPC/云原生科学计算引擎→返回科学解读与可视化报告，实现从需求到结果的全流程自动化。GaliLeo 已在理论计算化学、计算生物学、计算材料学、AIfor Science 与实验-计算交互等领域成功落地，其社区已有300+ 用户,涵盖70 余家高校与科研机构的硕博科研人员，对平台交互体验与计算效果高度认可。欢迎访问sidereus-ai.com，关注微信公众号“星使智算”或发送邮件至 sidereus_ai@163.com，共创科研智能体未来！

## 3.4.2ABACUS做不同真空方向计算的流程

本文算例的Cu(111)表面吸附CO结构在真空层方向的长度明显大于另外两个方向，我们会采用默认的单单点电子结构计算（ABACUS 中设置calculation为 scf）。之前的前两个测试案例和我个人对表面体系的计算经验表明，不同晶胞朝向对原子驰豫步数（即离子步步数）和优化收敛结构并没有影响，因而选用scf而非结构优化（原子驰豫）计算relax即可体现计算效率差距。

在此前的教程中，改变真空层的方向通过对 STRU 和KPT文件的三列进行手动轮换实现，但这一做法其实只适合对称性较高的晶胞，并且需要对分数坐标操作。这种晶胞朝向改变本质上是晶胞自然坐标系a,b,c方向的轮换，可以通过手写旋转矩阵的方式来实现，也可以通过已有程序完成，如ATOMKIT中的405功能可以便捷、可靠地完成这件事情，基于ASE也可以快速构建相关的代码。

在本教程中，Cu(111)表面吸附 CO 的结构由 ABACUS 智能体建模，ABACUS 计算所用的 INPUT 文件由ABACUS 智能体按照用户需求生成，KPT文件则通过 INPUT 设置的 kspacing 选项在 ABACUS 计算时自动生成，所用ABACUS 版本为LTS-v3.10.0，并基于202502版本的gnu-toolchain安装。计算采用ABACUS 赝势-轨道库APNS-PPORB-v1的推荐赝势及其对应效率基组完成。

## 结构建模和 STRU文件

本次测试的Cu(111)表面吸附CO结构由 ABACUS 智能体建模，我们让智能体首先构建了一个Cu晶胞，然后针对其(111)面进行切面，构造层数为4且进行了(2\*2)扩胞，一共有64个原子的Cu(111)表面，并添加15Angstrom的真空层，最后在切面上放置了一个CO分子。结构构建的对话过程如下：

<!-- image-->  
图1.构建Cu(111)-CO 表面吸附体系

构建完成后，智能体将结构保存为CIF文件，其结构如下图所示：

<!-- image-->  
图2.目标表面吸附结构可视化

在后续的智能体操作中，这一结构将被转化为ABACUS 的 STRU 结构文件。这一结构文件内容如下：

1 ATOMIC_SPECIES   
2 Cu 63.546000 Cu_0NCV_PBE-1.0.upf   
3 0 15.999400 0.upf   
4 C 12.010700 C.upf   
5   
6 NUMERICAL_ORBITAL   
7 Cu_gga_9au_150Ry_4s2p2d1f .orb   
8 O_gga_6au_100Ry_2s2p1d.orb   
9 C_gga_8au_100Ry_2s2p1d.orb   
10   
11 LATTICE_CONSTANT   
12 1.889726   
13   
14 LATTICE_VECTORS   
15 10.21062192033 0.00000000000 0.00000000000   
16 5.10531096017 8.84265797145 0.00000000000   
17 0.00000000000 0.00000000000 39.25270341532   
18   
19 ATOMIC_POSITIONS   
20 Cartesian

---

## 第 101–120 页

<table><tr><td rowspan=2 colspan=8>246425     0.00000000000     0.00000000000    15.00000000000 1 1 1</td></tr><tr><td rowspan=1 colspan=4>25     0.00000000000</td><td rowspan=1 colspan=2>0.00000000000</td><td rowspan=1 colspan=1>15.00000000000 1 1 1</td></tr><tr><td rowspan=1 colspan=4>26     1.27632774004</td><td rowspan=1 colspan=2>2.21066449286</td><td rowspan=1 colspan=1>15.00000000000 1 1 1</td><td rowspan=1 colspan=1>1 1 1</td></tr><tr><td rowspan=1 colspan=2>27</td><td rowspan=1 colspan=2>5.10531096017</td><td rowspan=1 colspan=2>4.42132898572</td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=2>28</td><td rowspan=1 colspan=2>3.82898322013</td><td rowspan=1 colspan=2>2.21066449286</td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=2>29</td><td rowspan=1 colspan=2>5.10531096017</td><td rowspan=1 colspan=2>2.94755265715</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=2>30</td><td rowspan=1 colspan=2>3.82898322013</td><td rowspan=1 colspan=2>0.73688816429</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>31</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>2.55265548008</td><td rowspan=1 colspan=2>2.94755265715</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1.27632774004</td><td rowspan=1 colspan=2>0.73688816429</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>2.55265548008</td><td rowspan=1 colspan=2>1.47377632858</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>3.82898322013</td><td rowspan=1 colspan=2>3.68444082144</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5.10531096017</td><td rowspan=1 colspan=2>1.47377632858</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.38163870021</td><td rowspan=1 colspan=2>3.68444082144</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=2>37</td><td rowspan=1 colspan=2>12.76327740042</td><td rowspan=1 colspan=2>4.42132898572</td><td rowspan=1 colspan=2>21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=2>38</td><td rowspan=1 colspan=2>11.48694966038</td><td rowspan=1 colspan=2>2.21066449286</td><td rowspan=1 colspan=2>21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=2>39</td><td rowspan=1 colspan=2>5.10531096017</td><td rowspan=1 colspan=2>4.42132898572</td><td rowspan=1 colspan=2>21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>3.82898322013</td><td rowspan=1 colspan=2>2.21066449286</td><td rowspan=1 colspan=2>21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=1>41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>2.55265548008</td><td rowspan=1 colspan=2>4.42132898572</td><td rowspan=1 colspan=2>15.00000000000 1 1 1</td></tr><tr><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.82898322013</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.63199347858</td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>1 11</td></tr><tr><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.55265548008</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.00000000000</td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>1 1 1</td></tr><tr><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.38163870021</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.63199347858</td><td rowspan=1 colspan=2>15.00000000000 1 1 1</td></tr><tr><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.65796644025</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7.36888164287</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.38163870021</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5.15821715001</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.10531096017</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7.36888164287</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.82898322013</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5.15821715001</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.10531096017</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5.89510531430</td><td rowspan=1 colspan=1>19.16846894355 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.38163870021</td><td rowspan=1 colspan=2>8.10576980716</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>51</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7.65796644025</td><td rowspan=1 colspan=2>5.89510531430</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8.93429418029</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.10576980716</td><td rowspan=1 colspan=2>19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.10531096017</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.84265797145</td><td rowspan=1 colspan=2>21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.82898322013</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.63199347858</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.65796644025</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.84265797145</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.38163870021</td><td rowspan=1 colspan=2>6.63199347858</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>5.10531096017</td><td rowspan=1 colspan=2>0.00000000000</td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.38163870021</td><td rowspan=1 colspan=2>2.21066449286</td><td rowspan=1 colspan=2>15.00000000000 111</td></tr><tr><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>10.21062192033</td><td rowspan=1 colspan=2>4.42132898572</td><td rowspan=1 colspan=2>15.00000000000 1 1 1</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.93429418029</td><td rowspan=1 colspan=1>2.21066449286</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15.00000000000 111</td><td rowspan=1 colspan=1>1 11</td></tr><tr><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>10.21062192033</td><td rowspan=1 colspan=1>2.94755265715</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.93429418029</td><td rowspan=1 colspan=2>0.73688816429</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7.65796644025</td><td rowspan=1 colspan=2>2.94755265715</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>6.38163870021</td><td rowspan=1 colspan=2>0.73688816429</td><td rowspan=1 colspan=2>17.08423447178 1 1 1</td></tr><tr><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7.65796644025</td><td rowspan=1 colspan=4>1.47377632858    19.16846894355 1 11</td></tr><tr><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>8.93429418029</td><td rowspan=1 colspan=4>3.68444082144    19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=3>10.21062192033</td><td rowspan=1 colspan=4>1.47377632858    19.16846894355 1 1 1</td></tr><tr><td rowspan=1 colspan=8>68    11.48694966038     3.68444082144    19.16846894355 1 1 1</td></tr></table>

<table><tr><td rowspan=1 colspan=1>69</td><td rowspan=1 colspan=6>7.65796644025     4.42132898572    21.25270341532 1 1 1</td></tr><tr><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=5>6.38163870021     2.21066449286    21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=4>10.21062192033     4.42132898572</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=4>8.93429418029     2.21066449286</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>73</td><td rowspan=1 colspan=6>7.65796644025     4.42132898572    15.00000000000 1 11</td></tr><tr><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=4>8.93429418029     6.63199347858</td><td rowspan=1 colspan=2>15.00000000000 1 11</td></tr><tr><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=2>7.65796644025</td><td rowspan=1 colspan=2>0.00000000000</td><td rowspan=1 colspan=1>15.00000000000 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=4>11.48694966038     6.63199347858</td><td rowspan=1 colspan=1>15.00000000000 1 11</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>77</td><td rowspan=1 colspan=4>12.76327740042     7.36888164287</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=3>11.48694966038</td><td rowspan=1 colspan=1>5.15821715001</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=3>10.21062192033</td><td rowspan=1 colspan=1>7.36888164287</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=3>8.93429418029</td><td rowspan=1 colspan=1>5.15821715001</td><td rowspan=1 colspan=1>17.08423447178 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=3>10.21062192033</td><td rowspan=1 colspan=1>5.89510531430</td><td rowspan=1 colspan=1>19.16846894355 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>11.48694966038</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>8.10576980716</td><td rowspan=1 colspan=1>19.16846894355 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=3>12.76327740042</td><td rowspan=1 colspan=1>5.89510531430</td><td rowspan=1 colspan=1>19.16846894355 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>84</td><td rowspan=1 colspan=4>14.03960514046     8.10576980716</td><td rowspan=1 colspan=1>19.16846894355 1 11</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=4>10.21062192033     8.84265797145</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=5>8.93429418029     6.63199347858    21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1>12.76327740042</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>8.84265797145</td><td rowspan=1 colspan=1>21.25270341532 1 1 1</td><td rowspan=1 colspan=1>111</td></tr><tr><td rowspan=1 colspan=1>88</td><td rowspan=5 colspan=6>88    11.48694966038     6.63199347858    21.25270341532 1 1 18990□910.00000092 193     7.65796644025     4.42132898572    24.25270341532 1 1 19495C960.00000097 198     7.65796644025     4.42132898572    23.102363415321 1 1</td></tr><tr><td rowspan=1 colspan=1>89</td></tr><tr><td rowspan=1 colspan=1>90□</td></tr><tr><td rowspan=1 colspan=1>92 1</td></tr><tr><td rowspan=1 colspan=1>93</td></tr></table>

智能体构建的表面吸附体系，其真空层（长边）方向在Z轴方向，这是符合表面吸附体系模拟的常规做法的。

这一表面结构不是四方晶系，而是六方晶系。在进行晶胞朝向调整时，不一定能用简单的 LAT-TICE_VECTORS 轮换来实现，而是需要对晶胞的自然坐标系进行旋转。同时，也不能对笛卡尔坐标进行轮换操作，而是要对分数坐标进行轮换操作。

通过与智能体对话，我们可以通过旋转晶格矢量，进一步构建不同真空层方向的这一表面体系。这一步骤也可以通过ATOMKIT等工具自行实现，相关对话如下：

<!-- image-->  
图4.将真空层旋转到X轴

<!-- image-->  
图3.将真空层旋转到Y轴

这些表面体系的LATTICE_VECTORS 是一个3×3矩阵，代表我们所计算体系在X,Y,Z三个方向上的晶格矢量a,b,c 大小，它们由LATTICE_CONSTANT（晶格常数）进行缩放，此处晶格常数为1.889726 Bohr,即1Angstrom。可以参考 htps:/abacus.deepmodeling.com/en/latest/advanced/input_files/stru.html#latice-vectors。它们的对比如下：

<!-- image-->

<table><tr><td>5</td><td>0.00000000000</td><td>0.00000000000</td><td>39.25270341532</td></tr><tr><td></td><td>6#真空层在Y 轴方向的 LATTICE_VECTORS</td><td></td><td></td></tr><tr><td>7</td><td>LATTICE_VECTORS</td><td></td><td></td></tr><tr><td>o</td><td>10.21062192033</td><td>0.00000000000</td><td>0.00000000000</td></tr><tr><td>9</td><td>0.00000000000</td><td>39.25270341532</td><td>0.00000000000</td></tr><tr><td>10</td><td>5.10531096017</td><td>0.00000000000</td><td>8.84265797145</td></tr><tr><td>11</td><td># 真空层在 X 轴方向的 LATTICE_VECTORS</td><td></td><td></td></tr><tr><td>12</td><td>LATTICE_VECTORS</td><td></td><td></td></tr><tr><td>13</td><td>39.25270341532</td><td>0.00000000000</td><td>0.00000000000</td></tr><tr><td>14</td><td>0.00000000000</td><td>10.21062192033</td><td>0.00000000000</td></tr><tr><td>15</td><td>0.00000000000</td><td>5.10531096017</td><td>8.84265797145</td></tr></table>

不难看出，虽然各个LATTICE_VECTORS的分量数值没有变化（这得益于六方晶系的对称性依然不低），但其位置不再是简单轮换。因而对于非四方晶系的体系，相关的晶胞朝向调整需要采用专门代码或者专门工具进行。

## INPUT文件

本次计算的 INPUT文件和 STRU 文件由ABACUS-Agent 基于CIF文件准备,准备方式如下所示：

基于对Cu_111_CO_adsorbate_swapped.cif，构建ABACUS计算任务，并设置kspacing为0.14

## ADAM

准备根据用户需求生成ABACUS输入文件，未经指定的计算关键词将被赋予默认值，这些默认值未必能给出高效良好的计算结果，请用户仔细检查生成的输入文件。

Structure files:['/share/data/adam-data-stage/workspace/2/1/475/Cu   
_111_CO_adsorbate_swapped.cif']   
Structure type: cif

## 显示更多>>

开始时间：2025-07-1513:51:50已耗时：20小时23分Token:17584

执行ABACUS输入文件生成器完毕

思考完毕开始时间：2025-07-15 13:52:15 已耗时：17秒Token:19164

## A ADAM

## 任务列表

## [√]构建Cu晶体

成功。构建好的晶体结构保存在：/share/data/adam-data-stage/workspace/2/1/475/Cu_bulk.cif。

[☑]基于构建好的Cu晶体，构建Cu(111)表面吸附CO分子的结构，表面层数为4，并做2x2扩胞，CO到表面距离为3埃，真空层厚度为15埃

·成功。构建好的表面吸附结构模型文件保存在：/share/data/adam-data-stage/workspace/2/1/475/Cu_111_CO_adsorbate.cif。

[√]对Cu_111_CO_adsorbate.cif做晶格变换，使之真空层的Z轴方向朝向Y轴

成功。晶格变换后的结构模型文件保存在：/share/data/adam-data-stage/workspace/2/1/475/Cu_111_CO_adsorbate_swapped.cif。

[√]对Cu_111_CO_adsorbate.cif做晶格变换，使之真空层的Z轴方向朝向X轴

成功。晶格变换后的结构模型文件保存在：/share/data/adam-data-stage/workspace/2/1/475/Cu_111_CO_adsorbate_swapped_to_X.cif。

[√]基于对Cu_111_CO_adsorbate.cif，构建ABACUS计算任务，并设置kspacing为0.14

·成功。生成的ABACUS输入文件INPUT和结构文件STRU所在位置为：./000000，INPUT文件内容为{'ecutwfc':100,'calculation':'scf','basis_type':'lcao','symmetry':1,'scf_thr':1e-07,'scf_nmax':100,'smearing_method':'gauss','smearing_sigma': 0.015,'mixing_type':'broyden','mixing_beta': 0.8,'ks_solver':'genelpa','kspacing':0.14}，请确认ABACUS输入文件是否符合要求，若是可开展ABACUS计算。

图5.准备ABACUS输入文件

从智能体的输出中我们可以提取出所用的INPUT。

INPUT_PARAMETERS   
2 ecutwfc 100   
3 calculation scf   
4 basis_type lcao   
5 kspacing 0.14   
6 symmetry 1   
7 scf_thr 1e-07   
8 scf_nmax 100   
9 smearing_method gauss   
10 smearing_sigma 0.015   
11 mixing_type broyden   
12 mixing_beta 0.8   
13 ks_solver genelpa

可以看到，INPUT文件中设置了calculation为scf，即单点电子结构计算，kspacing为0.14，这一值是用户的要求设置。其他参数均为ABACUSLTS 版本下开展 SCF计算的合理默认值。

## KPT文件

在开展计算时,ABACUS 会根据 INPUT文件中的 kspacing 参数自动生成 KPT文件。对于本次测试的Cu(111)表面吸附CO 结构，当真空层在Z轴方向时，生成的 KPT文件内容如下：

1 K_POINTS   
3 gamma   
4331000

需要注意的是，真空层方向的K点数目一般需要显式设置为1，以防止计算中混入不应存在的层间周期性。此处由kspacing参数自动生成的KPT文件是满足要求的。

## 3.4.3测试结果

测试计算由智能体调用计算资源完成，计算核数为16核，采用MPI方式并行，计算平台的CPU型号为Intel(R) Xeon(R)Platinum 9242 CPU @ 2.30GHz。测试结果由智能体统计。下表给出本次测试的总能量，电子步数和计算时间消耗，我们通过比较总能量和时间差异来判断真空层方向对计算结果的影响，可以看到，系统总能量和电子步迭代步数并不会受到不同真空层取向的影响，但计算时间会受到显著影响。真空层摆放在Z方向时，单电子步用时和总计算时间显著大于其他两个方向，达到了其他两个方向的三倍以上。

<table><tr><td rowspan=1 colspan=1>真空层朝向</td><td rowspan=1 colspan=1>系统总能量(eV)</td><td rowspan=1 colspan=1>计算总时间（s)</td><td rowspan=1 colspan=1>总电子步数</td><td rowspan=1 colspan=1>单电子步用时(s)</td></tr><tr><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>-318603.9704461048</td><td rowspan=1 colspan=1>1432.47</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>35.77 47.09</td></tr><tr><td rowspan=1 colspan=1>Y</td><td rowspan=1 colspan=1>-318603.9704469827</td><td rowspan=1 colspan=1>1421.25</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>35.37 46.94</td></tr><tr><td rowspan=1 colspan=1>Z</td><td rowspan=1 colspan=1>-318603.9704470374</td><td rowspan=1 colspan=1>4725.36</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>120.24 131.01</td></tr></table>

根据以上测试结果，建议读者们在进行表面和表面反应等含真空层体系的计算时，让真空层规避Z方向，并将表面的长边摆放在Z方向，以获得ABACUS 计算时的最大效率。同时，对于六方晶系或者类似于六方晶系的表面，则可将真空层摆放在Y方向，以获得最大的计算效率。

## 3.5模守恒赝势生成方法简介

## 3.5.1介绍

UPF（unified PPformat）是一种类似XML格式的文件格式，用于存储赝势及其相关参数。该格式由QuantumESPRESSO（QE）团队开发，并已成为许多量子计算软件包的标准格式之一。

ABACUS（截至v3.2.3）的Kohn-Sham密度泛函理论计算主要支持UPF格式的模守恒赝势，在使用ABACUS过程中，一般可以直接使用网上已经生成好的赝势，常见的下载位置：

ABACUS官网

SG15-ONCV，参考文献:Optimization algorithm for the generation of ONCV pseudopotentials

Quantum ESPRESSO 官网中的模守恒赝势

pseudo-dojo 官网（psp8 格式，需要转换）

PWmat官网

但是，在实际计算中，可能这些赝势并不符合当前的需求，这时候就需要自己生成模守恒赝势。可以通过其他格式赝势转换或者利用赝势软件生成两种方式来获得赝势。

注：ABACUS 的无轨道密度泛函理论计算（Orbital-Free Density Functional Theory，简称OFDFT）需要用到-种特殊的局域势（不包含非局域势)，一般采用 BLPS 局域赝势文件而非上文提到的UPF文件，如果做OFDFT计算的，可以阅读介绍OFDFT计算的文档，或者登录Emily A.Carter教授主页寻找相关的信息。

## 3.5.2从其他格式赝势转换

## 使用QEupflib 中的upfconv 进行转换

其支持将UPFv.1格式、vdb/van 格式（Vanderbilt US pseudopotential)、cpi/fhi格式（FHI/abinit）转化成 UPFv.2格式

```shell
1 path_to_QE/upflib/upfconv.x -u *.upf/UPF/vdb/van/cpi/fhi
```

注：这里的 path_to_QE 代表下载的Quantum Espresso 的软件包地址。

注：文件夹upflib在QE 5.x 和6.x 版本名称为upftools，在QE7.x版本为upflib。

## psp8格式转换

psp8 格式是ONCVPSP 软件生成的一种赝势格式，在Abinit 官网（pseudo-dojo）中，其使用的就是psp8的格式，目前没有直接将 psp8格式的鹰势直接转换成 UPF 格式的脚本，但可以将 psp8中输人文件部分摘抄下来,用ONCVPSP软件重新生成UPF格式的赝势（具体生成见下面ONCVPSP的讲解）。

可供参考的批量处理脚本：https://github.com/pipidog/ONCVPSP

## 3.5.3赝势的生成

下面介绍三个可以生成UPF 模守恒赝势的软件，分别是ONCVPSP，Opium，和ld1.x

## 1. ONCVPSP

## 介绍

ONCVPSP（Optimized Norm-Conserving Vanderbilt PSeudoPotential）是由 D.R.Hamann 等人提出的优化版模守恒赝势，其有更高的精度与效率。ONCVPSP依赖Libxc，支持多种交换关联泛函。

参考文献： Optimized norm-conserving Vanderbilt pseudopotentials 。

## 安装 11

安装Libxc  
Libxc 网址:Libxc - a library of exchange-correlation functionals for density-functional theory  
推荐下载 libxc-4.3.4  
然后执行如下命令：

1 cd libxc-4.3.4   
2 autoreconf -i   
3 ./configure --prefix='PATH/TO/LIBXC' CC=icc FC=ifort F77=ifort   
4 make   
5 make install

命令执行完毕后即可在PATH/TO/LIBXC目录下看到bin,include 和lib三个目录，代表安装成功

安装oncvpsp  
推荐下载 oncvpsp-4.0.1  
然后执行cd oncvpsp-4.0.1，进入文件夹  
这时需要修改make.inc 文件  
修改F77= ifort,F90=ifort,CC= icc  
FFLAGS 删去 -ffast-math  
LIBS改为 -qmkl="sequential"  
LIBS+和FFLAGS中的地址改为自己的PATH/TO/LIBXC  
具体如下:

```powershell
1 # System-dependent makefile options for ONCVPSP
2 # This must be carefully edited before executing "make" in src
3#
4 # Copyright (c) 1989-2019 by D. R. Hamann， Mat-Sim Research LLC and Rutgers
5 # University
6
7 ##### Edit the following lines to correspond to your compilers ####
8
9 F77 = ifort
10 F90 = ifort
11 CC = icc
12 FCCPP = cpp
13
14 FLINKER = \$ (F90)
15
16 #FCCPPFLAGS = -ansi -DLIBXC_VERSION=301 #Use this for versions <301
17 FCCPPFLAGS = -ansi -DLIBXC_VERSION=40O #Use this for versions >400
18
19 ##### Edit the following optimization flags for your system ####
20
21 FFLAGS = -03 -funroll-loops
22 CFLAGS = -03
23
24 ##### Edit the following LAPACK and BLAS library paths for your system ####
```

25   
26 LIBS = -qmkl="sequential"   
27   
28 ##### Edit the following for to use libxc if available #####   
29   
30 OBJS_LIBXC = exc_libxc_stub.o   
31   
32 # oncvpsp is compatible with libxc   
33 # To build oncvpsp with libxc， uncomment the e following lines and edit   
34 # the paths to point to your libxc library and include directories   
35 # make clean in src before rebuilding after changing this   
36   
37 LIBS += -L/PATH/TO/LIBXC/lib -lxcf90 -lxc   
38 FFLAGS += -I/PATH/TO/LIBXC/include   
39   
40 #LIBS += -L/home/drh/abinit/fallbacks/exports/lib -lxcf90 -lxc   
41 #FFLAGS += -I/home/drh/abinit/fallbacks/exports/include   
42   
43 OBJS_LIBXC = functionals.o exc_libxc.o

然后执行如下命令：

1 make  
2 # 如果 make-j 报错，可以忽略再 make，即可编译成功

安装测试完成后，即可在 src 目录下看到oncvpsp.x

可以在/.bashrc 中增加如下命令将oncvpsp.x加入环境变量，方便直接调用：

```shell
1 export PATH=\$PATH:/PATH/TO/ONCVPSP/srC
```

## 输人文件

输入文件的准备可以参考 PATH_TO_ONCVPSP/doc/32_Ge_annotated.dat，或者参考已知ONCV的展势里面的<PP_INPUTFILE>部分来写。

这里以铝（Al）为例，输入文件Al.dat:

```csv
1 # ATOM AND REFERENCE CONFIGURATION
2 # atsym，z, nc,nv, iexc psfile
3 Al 13144upf
4#
5 # n,1, f  (nc+nv lines)
6102
7 202
8 21 6
9 302
10 311
11 #
12# PSEUDOPOTENTIAL AND OPTIMIZATION
```

```csv
13 # lmax
14 1
15#
16 # l, rc,ep, ncon, nbas,qcut (lmax+1 lines, l's must be in order)
17 0 1.29163 0 5 8 10.3003
18 1 2.40653 0 5 8 7.02214
19 #
20# LOCAL POTENTIAL
21 # lloc，lpopt，rc （5），dvloc0
22 4 5 0.932267 0
23#
24 # VANDERBILT-KLEINMAN-BYLANDER PROJECTORS
25 # 1,nproj,debl (lmax+1 lines,l's in order)
26020
27120
28 #
29 # MODEL CORE CHARGE
30 # icmod, fcfact
3100
32#
33 # LOG DERIVATIVE ANALYSIS
34 # epsh1，epsh2，depsh
35 -5.0 3.0 0.02
36#
37 # OUTPUT GRID
38 # rlmax, drl
39 6.0 0.01
40 #
41 # TEST CONFIGURATIONS
42 # ncnf
430
44 # nvcnf
45# n 1 f
```

nc：c=core，芯电子层数

nv:v=valence，价电子层数，调整nc 和nv 可以改变碟势的价电子数

·iexc：生成厦势所用的交换关联泛男，详情可参考ONCVPSP-4.0.1目录下doc 内的 pwscf_exc.txt 文件，常用的如下

1 iexc==3 .or. iexc==-001009, 'functional="PZ"   
2 iexc==4 .or. iexc==-101130, 'functional="PBE"'

。 psfile：赝势格式，有upf 和 psp8 两种。ABACUS支持upf 的格式

·rc：rc 中较小的为赝势的截断半径。在温度较高或密度较大的温稠密体系，通常需要调整赝势的截断半径，推荐小于0.7倍维格纳半径，单位为Bohr；每个1对应一个rc，其中最小的为赝势的截断半径

·rc （5)：rc都要大于等于rc （5)。例如Al的1=0的rc为1.29163 Bohr，1=1的rc为2.40653 Bohr，rc（5)为0.932267Bohr，因此该赝势的截断半径为1.29163Bohr。

。 qcut：通过调整(+0.1)，直到推荐 ECUT达到最小

·dvloc0：通过调整(+0.5)，直到消除赝势的GHOST态（(可参考文献 Phys.Rev.B41,12264 (1990))

## 生成赝势

执行命令：

1 oncvpsp.x < Al.dat > Al.out   
2 sed -n '/<UPF version="2.0.1">/,/<\/UPF>/p' Al.out > Al.UPF

1 sh PATH_TO_ONCVPSP/scripts/run.sh Al

可以画出如下的图：

<!-- image-->  
图1.局域势函数与非局域势不同轨道角动量对应的半局域径向势函数

<!-- image-->  
图2.S赝波函数与全电子波函数对比

<!-- image-->  
图3.S的双投影波函数

<!-- image-->  
图4.不同能级波函数在截断半径处log导数对比，其影响散射性质的计算

<!-- image-->  
图5.不同轨道角动量对应的截断能

## 优化赝势

依次调节不同1对应的qcut（+-0.1)，检查对应ECUT（1）变化情况，直到 ECUT（1）达到最小参考的脚本：

```shell
1 !/bin/bash
2
3 qcut=17
4 dq=0.1
5 # 遍历 1=0 的 qcut 从 17 到 25，间隔为 0.1，输出其对应的 ECUT
6 while [ ‘echo "$qcut <= 25" 丨 bc' == "1" ]
7 do
8 cat > test.dat <<EOF
9# ATOM AND REFERENCE CONFIGURATION
10 # atsym z nc nv iexc psfile
11 Al 13.00 1 4 4 upf
12 #
13 # n 1 f energy (Ha)
14 1 0 2.00
15 2 0 2.00
16 2 1 6.00
17 3 0 2.00
18 3 1 1.00
19 #
20 # PSEUDOPOTENTIAL AND OPTIMIZATION
21 # lmax
22 1
23#
24# 1，rc, ep，ncon，nbas，qcut
25 0.68 -3.9750058$qcut
26 1 0.75 -2.55934 58 19.4
27 #
28 # LOCAL POTENTIAL
29 # lloc，lpopt， rc(5）， dvloc0
```

```shell
30 4 5 0.60 0.00000
31 #
32 # VANDERBILT-KLEINMAN-BYLANDER PROJECTORS
33 # 1，nproj，debl
34 0 23.69008
35 1 22.45967
36#
37 # MODEL CORE CHARGE
38 # icmod， fcfact
39 0•0.00000
40 #
41 # LOG DERIVATIVE ANALYSIS
42 # epsh1，epsh2，depsh
43 -5.00 3.00 0.02
44#
45 # OUTPUT GRID
46 # rlmax， drl
47 6.00 0.01
48 #
49 # TEST CONFIGURATIONS
50 # ncnf
51 0
52 # nvcnf
53#n 1 f
54E0F
55
56 ../../src/oncvpsp.x < test.dat > qcut$qcut.out
57 grep -A 5 "Energy error" qcut$qcut.out > _tmp.txt
58 E1=^sed -n "6p" _tmp.txt I avk '{print \$3}'`
59 E2=`sed -n "13p" _tmp.txt I awk '{print \$3}'`
60 E3=`sed -n "20p" _tmp.txt I avk '{print \$3}'`
61 E4='sed -n "27p" _tmp.txt | avk '{print \$3}'`
62
63 # echo $qcut $E1 $E2 $E3 $E4
64 if [‘echo "$E1>$E2" 丨 bc’ == '1' ] ;then
65 Emax1=$E1
66 else
67 Emax1=$E2
68 fi
69
70 if [‘echo "$E3>$E4" 丨 bc’ == '1' ];then
71 Emax2=$E3
72 else
73 Emax2=$E4
74 fi
75
76 echo $qcut $Emax1 $Emax2
77
```

78 qcut=\`echo "\$qcut+\$dq" I bc\`   
79 done

不同的1都会有对应的能量截断值 ECUT(I)。一般rc(l)越小，对应的 ECUT(I)越大，赝势也就越精确。由于DFT 计算中的截断能 ECUT是由赝势中较大的那个ECUT(I)决定，如果不同1的ECUT(I)相差很大，可以适当减小较小的ECUT(1）对应1的截断半径rc，使得不同1对应的ECUT(I)更接近，这样并不会增加赝势计算的ECUT，却可以提升精度。

这样调完之后如果有GHOST态，需要调整dvlocO（+-0.5）直到消除GHOST态，这个赝势才可使用

## 2. Opium

Opium 软件包可以生成RRKJ、TM或 Kerker径向波函数的赝势，官网有LDA、GGA赝势库（但是某些输入文件已经不匹配最新的4.1版本，需要稍作修改)

下载Opium之后利用configure 安装：

1 ./configure   
2 make

输人文件可以参考官网LDA、GGA 鹰势库，也可以参考 PATH_TO_OPIUM/tests，这里以铝（Al）为例，输人文件 al.param:

1 [Atom]   
2 Al   
36   
4 100 2.00 -13.0 # reference configuration   
5 200 2.00 -1.4 # reference configuration   
6 210 6.00 -0.4 # reference configuration   
7 300 2.00 -0.1 # reference configuration   
8 310 0.95 -13.0 # reference configuration   
9 320 -1.0 10.0 # 这条能带是非束缚态，需要将占据数调成负数，官网给错了   
10 [Pseudo]   
11 3 1.85 1.85 1.97   
12 opt   
13   
14 [Optinfo]   
15 7.07 10 # rc[a.u.] qc[sqrt (Ry) ] Nb   
16 7.07 10   
17 7.07 10   
18   
19 [Configs]   
203 # number of valence configurations   
21 #   
22 300 2.00 -13.0 # nlm occ eigen (- means auto-generate)   
23 310 1.00 -6.0   
24 320 0.00 -6.0   
25#   
26 300 1.00 -13.0   
27 310 2.00-6.0

这里的参数可以参考官网手册  
之后执行

opium al logall #输出在 log 文件里，依次执行全电子计算、计算赝势、计算非局域势、可移植性测试  
2 opium al log upf # 生成 al.upf 的输出文件  
3 upfconv.x -u al.upf # 利用 QE 的 upflib 将 UPFv.1 转化成 UPFv.2 格式

## 3. ld1.x

QE 的 atomic 模块中的ld1.x支持生成赝势。其不仅可以生成模守恒赝势、还支持超软赝势、PAW方法，支持全相对论、标量（非）相对论赝势（rel,non-rel/sca-rel)，其径向波函数支持TM（更稳定）与RRKJ两种方法,支持交换关联近似7类LDA（pz）,GGA（pbe,pbesol,revpbe,bp,wc,pw91）。

下载好QE 软件后，可以直接安装：

1 ./configure   
2 make ld1

## pslibrary 赝势库

推荐ld1.x 生成其自带的pslibrary赝势库，下载好 pslibrary.1.0.0.tar.gz，解压，进入文件夹，修改 QE_path 文件，指定QE 的路径。然后打开make_ps文件，解锁相应赝势：

1# These two files generate PAW and US PPs for all elements. These are   
2 #high accuracy - high kinetic energy cut-off PPs.   
3#   
4．../paw_ps_high.job # 默认打开的   
5. ../us_ps_high.job   
6#   
7# These two files generate additional PAW and US PPs for some elements.   
8# These are less accurate PP than the previous one but require   
9# lower kinetic energy cut-off or have less projectors or less semicore   
10 # states.   
11 #   
12 . ../paw_ps_low.job   
13 . ../us_ps_low.job   
14

15 # Uncomment the following line to generate the old pslibrary O.3.1 PPs.   
16 ＃打开下面的注释，可以解锁其余赝势   
17 #   
18 #．../paw_ps_collection.job# 可以打开注释   
19 #. ../us_ps_collection.job ＃ 可以打开注释   
20   
21 # Uncomment the following line to generate the NC-PPs. Be very careful   
22 # these PPs are completely untested.   
23 #虽然可以生成，但其也提示该赝势的准确性并不能保证   
24 #   
25 #. ../nc_ps_collection.job # 可以打开注释

## 下面运行

```batch
1 ./make_all_ps
```

就能看到在一个个生成赝势库里的赝势：

1 Making Ac.pz-spfn-kjpaw_psl.1.0.0.in ... Done   
2 Making Ac.pz-spfn-rrkjus_psl.1.0.0.in ... Done   
3 Making Ag.pz-n-kjpaw_psl.1.0.0.in ... Done   
4 Making Ag.pz-n-rrkjus_psl.1.0.0.in... Done   
5 Making Ag.pz-spn-kjpaw_psl.1.0.0.in ... Done   
6 Making Ag.pz-spn-rrkjus_psl.1.0.0.in ... Done   
7 Making Al.pz-n-kjpaw_psl.1.0.0.in ... Done   
8 Making Al.pz-n-rrkjus_psl.1.0.0.in ... Done   
9 Making Al.pz-nl-kjpaw_psl.1.0.0.in ... Done   
10 Making Al.pz-nl-rrkjus_psl.1.0.0.in ... Done

如果不想生成所有的元素的赝势，则可以修改make_ps 的element，例如：

1 element='C Si Ge'

## 生成自己的赝势

参数的详细解释见PATH_TO_QE/atomic/Doc/INPUT_LD1.html 或线上的文档（不是最新的，建议前者），输入文件可以参考pslibrary 赝势库的例子。

这里就简单给个Al（铝）的例子，准备输入文件al.in:

1 &input   
2 title='Al',   
3 zed=13.0,   
4 rel=1,   
5 config='[Ne] 3s2 3p1 3d-2.0 4f-2.0',   
6 iswitch=3,   
dft='PBE'   
8 /   
9 &inputp

10 pseudotype=2,   
11 file_pseudopw='Al.pbe-n-nc.UPF'，# 输出赝势   
12 1loc=2,   
13 nlcc=.true.,   
14 tm=.true.   
15/   
16 3   
17 3S 10 2.00 0.00 2.60 2.60 0.0   
18 3P 2 1 1.00 0.00 2.60 2.60 0.0   
19 3D 3 2 0.00 0.10 2.60 2.60 0.0

## 运行

1 ld1.x < al.in  
即可生成 Al.pbe-n-nc.UPF 模守恒赝势。

## 3.5.4结尾

ABACUS 使用的是模守恒赝势，基于模守恒赝势还可以产生数值原子轨道，进行基于数值原子轨道的高效率密度泛函理论计算。有些情况下，网上提供的赝势不能满足需求，这个时候赝势的生成需要对赝势理论有比较深入的了解之后，才能调整好相关参数，生成质量较好的赝势。此外，生成之后，应该经过较为充分的测试，才能保证赝势的可移植性和正确性。如果大家有任何问题，欢迎发email至文档开头的邮箱。

## 3.6数值原子轨道（一)：ABACUS中的数值原子轨道命名和使用方法

## 3.6.1 数值原子轨道的背景知识

## 数值原子轨道

对电子结构的求解可以在不同的表象下进行，电子的波函数以及势函数也可以在不同基矢量下表示，常用的有平面波和局域轨道等。LCAO（Linear Combinationof Atomic Orbitals）方法就是通过局域的原子轨道来求解量子力学问题。其中局域轨道的选取方式有多种，例如高斯轨道（Gaussian type Orbitals，GTOs 或 Gaussians）、数值原子轨道（Numerical Atomic Orbitals）、万尼尔函数（Wannier Functions）等。

## 数值原子轨道作为基矢量的优点

数值原子轨道作为基矢量有几个优点：第一，基矢量个数相比于一些常用的基矢量（例如平面波和实空间网格）大幅度降低；第二，数值原子轨道是局域的，空间上可以严格截断，采用数值原子轨道来构建体系的哈密顿量的效率可以达到线性标度的时间复杂度。

## 构造数值原子轨道基组的几种方案

构造精度高、可系统提升数量、可移植性好的原子轨道基组颇有挑战，因此也有多种方案被提出。例如,Junquera 等人提出在一维薛定谔方程中加人不同形式的约束势场，从而求解出具有严格截断的数值原子轨道[1]。Ozaki在OpenMX软件中采用变分的方法来优化局域轨道的形状，从而得出一组最优的数值原子轨道[2]。Volker等人提出在一个大的局域轨道基组中挑选最合适的局域轨道组成不同等级的基组轨道，该方案用于全电子密度泛函理论软件FHI-aims 中[3]。Chen 等人提出利用前人提出的溢出函数（Spillage function）[4-5]来构造可系统提高数量的数值原子轨道，其中每个轨道都由一组球贝塞尔函数作为基矢量展开，该轨道被用在ABACUS（原子算筹）软件中[6]。

## 3.6.2 数值原子轨道的命名方法

## 数值原子轨道的组成

数值原子轨道（Numerical Atomic Orbitals，简称NAO）是目前在 ABACUS 程序中支持的一种基矢量。从数学形式上来看，数值原子轨道可以分解为径向函数fi、和球谐函数 $Y _ { l } m$ 的乘积。

$$
\phi _ { l m \zeta } ( \mathbf { r } ) = f _ { l } \zeta ( r ) Y _ { l m } ( \hat { r } ) ,
$$

其中1是角量子数，m是磁量子数，（代表了每个角量子数上对应的多个径向轨道，实际计算中通常采用多于1个轨道来增加基矢量的完备性。

## 数值原子轨道的命名方法

数值原子轨道有一套常用的命名方案用来表示选取的基组大小，早期该命名方案在 SIESTA软件中被采用[7][8][9][10]，后来在OpenMX、FHI-aims 和ABACUS 中也采取这套命名方案。具体来说是，对于每个被电子占据的角量子数1，若采用1条径向轨道，则称该基组为 Single-(轨道，简称 SZ轨道基组。若采用2条径向轨道，则称该基组为Double-( 轨道，简称DZ轨道基组。

目前，在许多赝势结合数值原子轨道的程序里，通常会在DZ轨道的基础上引人1条极化（polar）的径向轨道，即角量子数更高的轨道，来组成 DZP（Double-[ valence orbitals plus SZ polarization）轨道基组。此外，还有基组数量更大的 TZDP（Triple-( valence orbitals plus DZ polarization）轨道等。

## 数值原子轨道基组的个数

数值原子轨道基组的具体个数除了取决于轨道类别（比如 SZ、DZP、TZDP）外，也取决于元素种类、所选取的赝势。

轨道类别：比如对于O元素的赝势而言，一般将两个1s 电子作为核内电子，在构造赝势的时候只考虑它的外层6个价电子部分2s、2p轨道，则该O赝势下的电子的极化轨道为d轨道。故它的SZ（Single-c）轨道包含1组s轨道、1组p轨道，共 $1 ^ { * } 1 + 1 ^ { * } 3 = 4$ 个轨道；DZP轨道包含2组s轨道、2组p轨道、1组d轨道，共包含$2 ^ { * } 1 + 2 ^ { * } 3 + 1 ^ { * } 5 = 1 3$ 个轨道；TZDP轨道包含3组s轨道、3组p轨道、2组d轨道，共包含 $3 ^ { * } 1 + 3 ^ { * } 3 + 2 ^ { * } 5 = 2 2$ 个轨道。

元素种类：比如同为DZP轨道，对于H元素（1个价电子）即为2组s轨道、1组p轨道（p轨道也是H的极化轨道），共包含 $2 ^ { * } 1 + 1 ^ { * } 3 = 5$ 个轨道。对于O元素即为2组s轨道、2组p轨道、1组d轨道，共包含$2 ^ { * } 1 + 2 ^ { * } 3 + 1 ^ { * } 5 = 1 3$ 个轨道。

赝势：比如对于Fe元素而言，同为DZP轨道。若选取的赝势为,

$$
\mathrm { [ N e ] 3 } s ^ { 2 } 3 p ^ { 6 } 4 s ^ { 2 } 3 d ^ { 6 }
$$

即

$$
[ \mathrm { N e } ] = 1 s ^ { 2 } 2 s ^ { 2 } 2 p ^ { 6 }
$$

部分为芯电子, $3 s ^ { 2 } 3 p ^ { 6 } 4 s ^ { 2 } 3 d ^ { 6 }$

部分为16个价电子，则DZP轨道包含2\*2套s轨道（3s和4s)、2套p轨道(3p)、2套d轨道(3d)、1套f轨道(极化轨道)，即每个原子的DZP数值原子轨道个数为

$$
2 ^ { * } 2 ^ { * } 1 + 2 ^ { * } 3 + 2 ^ { * } 5 + 1 ^ { * } 7 = 2 7
$$

个轨道。若选取的赝势为

$$
[ \mathrm { A r } ] 4 s ^ { 2 } 3 d ^ { 6 }
$$

则DZP轨道包含2套s轨道、2套d轨道、1套p轨道，共 $2 ^ { * } 1 + 2 ^ { * } 5 + 1 ^ { * } 3 = 1 5$ 个轨道。

## 3.6.3ABACUS中数值原子轨道的使用方法

## ABACUS中的数值原子轨道文件

ABACUS 提供了已经生成好的数值原子轨道库打包文件供下载（官网下载链接），对于绝大多数计算任务，这些数值原子轨道是经过精度和可靠性验证，可以直接使用的。这些轨道文件的开头提供了关于该轨道的重要信息。以O_gga_7au_100Ry_2s2pld.orb轨道文件为例，首先文件名包含的信息有：氧元素(o),GGA泛函（gga)，数值原子轨道截断半径（7au，即7Bohr)，推荐能量截断值（100Ry)，数值原子轨道个数（2s2pld，2个s的径向轨道，2个p的径向轨道，1个d的径向轨道)。值得注意的三个点是：

ABACUS 里用到的模守恒赝势和轨道是需要匹配的。因为不同的赝势可能有不同的价电子，而数值原子轨道是用来描述这些价电子的，如果赝势的价电子多，则相应的默认数值原子轨道基组数量也会增多。

推荐用户直接使用轨道文件包含的能量截断值做计算。ABACUS 里的数值原子轨道在生成时,是尽量的去匹配平面波输出的波函数，从而优化得到的。而平面波计算是有一个能量截断值的，因此为了尽可能的保证LCAO的精度，我们建议直接使用推荐的能量截断值，而不需要真正做计算的时候做体系能量随着能量截断值变化的收敛性测试。当你在LCAO计算时改变能量截断值时候，其实只是改变了平面波的个数，这些平面波是用来做一些数值计算的，本质上并没有改变基矢量的个数，而基矢量的个数增加是靠改变基组大小，例如DZP到TZDP来实现的。

特殊情况可以自己生成数值原子轨道。如果用户有自己特殊的赝势，或者对目前提供的原子轨道的参数或者个数感觉不满意，例如对于导带的描述能力较差，也可以自己生成数值原子轨道，具体生成的方法可以参考这个系列文档的第二和第三篇。打开以上提到的氧的数值原子轨道文件，文件的开头如下：

2 Element 口   
3 Energy Cutoff(Ry) 100   
4 Radius Cutoff(a.u.) 7   
5 Lmax 2   
6 Number of Sorbital--> 2   
7 Number of Porbital--> 2   
8 Number of Dorbital--> 1   
9 =   
10 SUMMARY END

这里包括元素种类（Element)，生成轨道时指定的截断能量值（Energy Cutoff）、截断半径（Radius Cutoff)，最大角量子数（Lmax）及各角量子数轨道的个数（Numberof \*orbital，\*为S、P、D等轨道角动量)。根据这些信息可以知道该轨道的类别，比如该举例文件即包含O的DZP轨道。

上述 ABACUS 提供的数值原子轨道文件中，在取名中即包含文件开头的重要信息，包括各角量子数的轨道个数。由这些轨道的文件名即可判断轨道类型。

轨道文件中包含的后续内容为不同类型、不同角量子数（L）的多个径向数值原子轨道（即不同N）的具体数据，目前轨道是存在均匀格点上的，之后有可能也会支持非均匀格点。

---

## 第 121–140 页

## 如何选择数值原子轨道文件

用户需要根据精度要求，选择数值原子轨道合适的截断能量值、截断半径及轨道类型。截断能量值、截断半径越大，轨道类型提升（SZ、DZP、TZDP)，精度越高，结果更接近平面波。注意对于不同元素，要达到同样的精度，以上数值设置并不一定相同。

## 如何修改数值原子轨道文件的设置，获得低精度轨道类型文件

在 ABACUS 中生成数值原子轨道文件时，可以设置同时生成多种精度的轨道文件，即设置 SIAB_INPUT 文件中的参数 Save Orbitals，详见篇目（三)。

如果已有一份数值原子轨道文件，但希望直接用此文件进行低精度计算，比如希望使用DZP 轨道文件进行SZ 计算。可以修改文件开头的信息实现，具体是Lmax及各角量子数轨道个数，将这些参数调整到低精度计算对应的数值。文件的后续内容不需要做更改。

比如对于O元素而言，DZP 轨道文件参数如3.1所示，若要进行 SZ 基矢量的计算，则将Lmax 设置为1,Number of Sorbital/Porbital/Dorbital=1/1/O 即可。

## 3.6.4参考文献

[1] J. Junquera, O.Paz,D.Sanchez-Portal,and E.Artacho,Numerical Atomic Orbitals forLinear-Scaling Calculations,Phys.Rev.B,64,235111 (2001).

[2] T. Ozaki, Variationally Optimized Atomic Orbitals for Large-Scale Electronic Structures, Phys.Rev. B,67, 155108 (2003).

[3] V. Blum,R. Gehrke,F. Hanke,P. Havu, V. Havu,X.Ren,K. Reuter, and M. Schefler,Ab Initio Molecular Simulations with Numeric Atom-Centered Orbitals, Comput. Phys. Commun.,180, 2175 (2009).

[4] M. Chen, G.-C. Guo,and L.He,Systematicaly Improvable Optimized Atomic Basis Sets for Ab Initio Calculations,J. Phys.: Condens.Matter, 22, 445501 (2010).

[5]M.Chen, G.-C.Guo,andL.He,Electronic Structure Interpolation via Atomic Orbitals,J.Phys.: Condens. Mater, 23, 325501 (2011).

[6]Liu Xiao-Hui et al., Introduction to first-principles simulation package ABACUS based on systematically improvable atomic orbitals,Acta Phys. Sin., 64,187104 (2015).

[7] Sanchez-PortalD,Ordejon P,ArtachoE,et al. Density-functional method for verylarge systems withLCAO basis sets,J. International Journal of Quantum Chemistry, 65, 453-461 (1997).

[8] ArtachoE,Sänchez-Portal D,Ordejon P,etal. Linear-scaling ab-initiocalculations for large and complex systems, J. Physica Status Solidi (b),215, 809-817 (1999).

[9] Junquera J,Paz O,Sanchez-Portal D,et al. Numerical atomic orbitals forlinear-scaling calculations,J.Physical Review B,64,235111 (2001).

[10] Soler JM,Artacho E, Gale JD,et al. The SIESTA method for ab initio order-N materials simulation,J. Phys.: Condens.Matter,14,2475 (2002).

## 3.7 数值原子轨道(二)：生成给定模守恒赝势的数值原子轨道

## 3.7.1 ABACUS中的数值原子轨道背景知识

## 数值原子轨道

ABACUS中的三维数值原子轨道[1]：可以分解为径向部分与角向球谐函数的乘积，即

$$
\phi _ { \mu } ( r ) = f _ { \mu , l } ( r ) Y _ { l m } ( \hat { r } ) ,
$$

其中 $\mu = \{ \alpha , i , \xi , l , m \}$ ，α是原子种类的指标，i是每个原子种类对应的原子个数的指标， $\xi$ 是角动量1对应的径向函数的个数，m 是磁量子数。

径向部分函数：第 $\mu$ 个轨道的径向部分函数可以写为球贝塞尔函数的线性组合 (可参考文献[1])

$$
f _ { \mu l } ( r ) = \left\{ \begin{array} { l l } { \sum _ { q } c _ { \mu q } j _ { l } ( q r ) , } & { r < r _ { c } } \\ { 0 , } & { r \geq r _ { c } } \end{array} \right. ,
$$

其中 ${ \dot { \jmath } } _ { l } ( q r )$ 是波矢为q的球贝塞尔函数， $r _ { c }$ 是径向部分的截断半径，满足关系 $j _ { l } ( q r _ { c } ) = 0 \sb { \circ } \textnormal { q }$ 的个数由能量截断值决定(以下会介绍)。在实际计算里，通过用户指定每种元素的数值原子轨道截断半径和能量截断值来确定需要用到多少个球贝塞尔函数。组合系数 $c _ { \mu q }$ 是可优化的量，决定了数值原子轨道的形状。

## 参考体系

参考系统：在ABACUS 中构造某个元素对应的数值原子轨道的最普遍方式是为每个元素选取一组参考体系做平面波计算（可以是单k点也可以是多k点)，得到这些参考体系的电子波函数，然后通过最小化自定义的一个溢出函数（下面会介绍）来优化 $c _ { \mu q }$ 系数。换句话说，我们需要选定一组参考系统，生成数值原子轨道基组，并期望它们有良好的迁移性，以便可以在更一般的情况下使用。

Dimer（二聚体)：我们发现二聚体是较为合适的参考系统。具体来说，推荐选择的是若干个不同键长的二聚体（dimer），有时也可选择三聚体（trimer）或四聚体。对同一个元素，我们通常选择几个键长不同的二聚体（例如 5-6个），这些键长覆盖二聚体稳定键长附近能量区间，通常可以取与最稳定能量差别为1-2个eV的能量曲线上的点。

周期性体系：数值原子轨道还有一些特殊用途，例如能带插值。面对这种场景，可以选择一系列带多个k点的周期性体系，此时参考态中的每个原子构型就对应了不同k点指标和不同能带指标的波函数，这些波函数会被用于最小化溢出函数来生成数值原子轨道，可以参考文献[2]。

## 溢出函数

溢出函数（英文为 spillage)：由给定参考体系的多个电子波函数与生成的数值原子轨道之间的差别来定义。换个角度理解，溢出函数的大小反映的是由一组”精确”波函数所张成的希尔伯特空间与局域轨道所张成的空间的差别。溢出函数定义如下：

$$
S = \frac { 1 } { N _ { n } } \sum _ { n = 1 } ^ { N _ { n } } \left. \Psi _ { n } | 1 - \hat { P } | \Psi _ { n } \right. ,
$$

其中 $\Psi _ { n }$ 表示平面波基组下的本征态， $N _ { n }$ 是选定的参考电子态的个数， $\hat { P }$ 是由所有原子轨道张成的投影子

$$
\hat { P } = \sum _ { \mu v } | \phi _ { \mu } \rangle S _ { \mu v } ^ { - 1 } \langle \phi _ { v } | ,
$$

其中 $S _ { \mu v } = \langle \phi _ { \mu } | \phi _ { v } \rangle$ 是局域轨道 $\phi _ { \mu }$ 与 $\phi _ { v }$ 之间的重叠矩阵，投影子用到了重叠矩阵的逆。

轨道正交：事实上，如果我们有了给定的数值原子轨道，我们可以在此基础上再次定义新的溢出函数来得到一组新的数值原子轨道，这也是为什么该算法可以系统性的增加数值原子轨道的原因。定义的方式也比较直接，可以将 $\left| { \Psi _ { n } } \right.$ 和已有原子轨道形成的投影子 $\hat { P }$ 正交，公式为

$$
\left| \Psi _ { n } ^ { n e w } \right. = \left( 1 - \hat { P } \right) \left| \Psi _ { n } \right. ,
$$

这个公式里，左边的波函数经过正交已经包含原子轨道的成分，这样可以再次作为新的参考体系波函数生成新的数值原子轨道，以上算法已经被实现在SIAB这个程序里。

最小化溢出函数：优化原子轨道时，最终溢出函数值定义为选定的几个二聚体的平均溢出函数值。溢出函数可以用来评估获得的原子轨道基组的质量，溢出函数越小，则原子轨道所张成的空间越接近于参考态波函数。

在 ABACUS中，通过最小化溢出函数的方式来优化 $c _ { \mu q }$ ，该系数决定了原子轨道的径向函数的形状。这种方法的好处在于

·轨道形状可调：可以由球贝塞尔函数基组的线性组合自动生成，而不需要任何先决条件，因此原则上可获得完全优化的原子轨道。

·基组数量可调：可以用统一的方式系统的增加轨道数量，包括增加径向轨道的数量（multi-C）和增加更高角动量的轨道（极化轨道）两种方式。为此，我们可以首先生成具有给定角动量的轨道，然后更高级的轨道可以使用相同的程序生成，但是此时只需最小化剩余的希尔伯特空间中的溢出函数，该空间与所有先前生成的原子轨道所张成的空间正交。

·参考系统可调：通过选择参考系统及对应能级，可以提高原子轨道基组在不同环境的迁移性。例如，对于涉及到激发态能级的计算，往往需要在生成数值原子轨道的时候，考虑这组轨道对于非占据态能级的拟合能力，这个时候可以在生成参考体系的时候加入一定数量的非占据能级。

## 模拟退火算法 SA (Simulated Annealing)

我们首先介绍模拟退火法 Simulated Amnealing（SA）来最小化溢出函数[1]，从而优化球贝塞尔函数的系数$c _ { \mu q }$ 。在模拟退火算法中我们用的是Metropolis 的算法，需要设置一个初始温度，并且指定温度下降的方式。代码部分我们之后会介绍，这里先简要介绍一下算法。

初始系数：如果没有特别指定，一开始的系数 $c _ { \mu q }$ 是随机产生的，每一步SA会更新其中一个系数，当溢出函数被改变的数值在允许接收的范围内时，参数 $c _ { \mu q }$ 会被更新。

优化过程：当要优化的所有轨道的所有参数 $c _ { \mu q }$ 都被更新过一次之后，我们称为一步优化。我们发现大约在每个温度行走 500～1000 步就已经足够产生好的原子轨道。最开始的温度我们一般取为1.0e-5，事实上我们发现即便初始温度取得更小，最后的结果也是不错的。在每一步优化之后，我们将温度乘上一个比率，来达到降温的目的从而得到新的温度，这个参数我们一般取为0.50.8，最后的温度一般降到1.0e-91.0e-12。接收的概率可以控制在0.20.4 范围内。

能量截断值：这里的能量截断值实际上是能量截断值，我们知道每个平面波eqr都对应一个二阶求导后的动能 $q ^ { 2 } / 2$ 。同样的，每个球贝塞尔函数 $j _ { i } ( q r )$ 也有一个对应的动能是 $q ^ { 2 } / 2$ 。在平面波计算里我们通过能量截断值来控制平面波的数量，从而控制计算精度。在生成数值原子轨道的时候，我们也通过能量截断值来控制球贝塞尔函数的个数，从而控制生成的数值原子轨道的精度。

对于 ABACUS 的平面波计算和产生球贝塞尔函数ji(qr)个数的计算，我们建议采用同样的能量截断值，该截断值取得越高，基矢量个数越多同时越组也越完备。对于采用腰势的平面波计算，1530个球贝塞尔函数已经足够得到质量比较高的数值原子轨道。

数值原子轨道的二阶可导：除了以上操作之外，还有一个需要注意的地方。为了使数值原子轨道的动能积分是定义的，需要使原子轨道的二阶导数连续。这可以通过将原子轨道的径向部分乘以一个平滑函数g(r)来实现

$$
g ( r ) = 1 - \exp \left[ - \frac { ( r - r _ { c } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right] ,
$$

该径向轨道会在数值原子轨道的截断值以内起作用，使其在截断值处2阶连续，超过阶段半径之后就严格截断。在我们的测试中，我们发现参数o对轨道的质量影响不大，所以我们一般固定o为常数0.1。

数值原子轨道动能和再次使用模拟退火算法：如果仅仅最小化溢出函数值，原子轨道的径向部分经常会出现震荡，原因是震荡有时有利于溢出函数的值尽可能的低，但震荡厉害的函数往往能高，可移植性较差。因此，为了去除这些实际非物理的震荡，在算法中我们还定义了原子轨道的动能：

$$
T _ { \mu } = \sum _ { q } c _ { \mu q } ^ { 2 } q ^ { 2 } / 2 + k ,
$$

并再次通过模拟退火算法最小化原子轨道的动能，其中k 则是模拟退火的一个惩罚函数：

$$
k = \left\{ \begin{array} { l l } { 0 , } & { S / S _ { 0 } - 1 < \Delta } \\ { \infty , } & { S / S _ { 0 } - 1 > \Delta } \end{array} , \right.
$$

这里 $S _ { 0 }$ 是在优化动能之前所得到的数值原子轨道算出来的溢出函数的值，而S 则是当前在参数 $c _ { \mu q }$ 下面的溢出函数的值。我们发现参数 $\Delta = 0 . 0 0 2 \sim 0 . 0 0 5$ 足够使得原子轨道平滑了。轨道动能也是通过同样的模拟退火算法得到，经过动能优化之后，轨道的形状会变平滑，同时溢出函数的值也会相比较有可能优化的轨道会有一点增加，但是对精度的影响是很小的。

优化动能用的模拟退火算法几乎和优化溢出函数一样，除了一点点参数的区别。目标函数选择的是所有轨道中最大的动能，而不是像溢出函数一样取平均值。对于每个轨道来说，温度是不一样的，因为每个轨道会有一个不同的动能，在这步动能优化中，我们发现结果对初始的温度不敏感，但最终达到的温度必须取得足够小。实际中我们发现只需要100200步，动能就可以下降得很快。所以一般来说优化动能的步数就取得比优化溢出函数要少。

## 3.7.2ABACUS中产生数值原子轨道的具体流程

注：本文档介绍的是模拟退火算法 SA，不适用于基于Pytorch的PyTorch-Gradient方法，关于这两个算法，可以参考本数值原子轨道中文文档系列的第三篇文档。

## 平面波计算

简化计算量：前文我们提到的溢出函数公式，如果把投影子 $\hat { P }$ 带进公式，则可以发现我们不需要在计算溢出函数的时候显示的包含任何基矢量（例如平面波）的操作。实际上，我们只需要先算好数值原子轨道 $\left| \phi _ { \mu } \right.$ 和电子波函数 $\left| u _ { i } \right.$ 的重叠矩阵（OverlapMatrix） $\big ( u _ { i } | \phi _ { \mu } \big )$ 并存下来，以及数值原子轨道之间的重叠矩阵 $\langle \phi _ { \mu } | \phi _ { \nu } \rangle$ 存下来就可以算溢出函数了。更进一步，我们意识到数值原子轨道 $\mathinner { | { \phi _ { \mu } } \rangle }$ 的径向部分在程序里是写成了球贝塞尔函数的线性组合，根据线性叠加的原理，我们只需要计算球贝塞尔函数和电子波函数的内积，以及球贝塞尔函数之间的内积，并把这些结果存下来，就可以优化系数 $c _ { \mu q }$ 了！注意这里指标准遍历体系的所有电子态，包括所有的k点以及能带对应的电子波函数。

换句话说，只需要存储线性代数里的内积结果而不是每个波函数在基矢量下的展开系数就可以优化我们想要的系数，因此数值原子轨道的优化过程可以极大提速！

采用ABACUS做平面波计算：

因此，第一步我们先调用 ABACUS 主程序，计算出以上overlap 矩阵，首先需要准备 INPUT文件如下：

1 INPUT_PARAMETERS   
2 suffix H   
3 pseudo_dir /home/liuyu/github/abacus-develop/tests/PP_ORB   
4 wannier_card INPUTw   
5 calculation scf   
6 ks_solver dav   
7 nspin 1   
8 lmaxmax 1   
9   
10 symmetry 0   
11 nbands 10   
12   
13 ecutwfc 100   
14 scf_thr 1.0e-7   
15 scf_nmax 1000

16   
17 smearing_method gauss   
18 smearing_sigma 0.01   
19   
20 mixing_type pulay   
21 mixing_beta 0.4   
22 mixing_ndim 8   
23 printe 1   
24   
25 bessel_nao_ecut 100 // same as ecutwfc   
26 bessel_nao_rcut 6 // rcut

如上所示，该INPUT与平常的平面波计算相比，只有3个参数（已标红）需要注意：

wannier_card：指定wannier相关参数的输入文件名INPUTw

·bessel_nao_ecut：球贝塞尔函数的截断能，根据公式 $\sqrt { E _ { c u t } } \times r _ { c u t } / \pi$ 来决定求贝塞尔函数的数量

bessel_nao_rcut：数值原子轨道的截断半径 $r _ { c u t }$

此外，我们还需要准备输入文件 INPUTw如下（将来计划会把 INPUTw文件去掉，把参数加人INPUT里)：

1 WANNIER_PARAMETERS   
2 out_spillage 2 // output overlap   
3 spillage_outdir OUT.H// could be the same as OUT.\\${suffix}

通过ABACUS 做平面波基矢量计算可以计算出：

```javascript
1.电子波函数{亚}和球贝塞尔函数 j(qr)之间的overlap(亚ilj(qr));
```

2.球贝塞尔函数之间的overlap jtz(q2r)ljt(qr)>;

3．电子波函数{亚}之间的overlap{亚|亚>。

这些输出将存在文件our.f(sufix)/orb_matrix.o.dat中，而这个文件将被读入进行数值原子轨道的生成。

## 编译 SIAB

SIAB 的全称是 Systematically Improvable Atomic orbital Basis generator based on spillage formula，目前是包含在 ABACUS 软件里的一个独立程序，可用来读入以上ABACUS 产生的数据文件并且生成数值原子轨道，具体编译方法如下：

拉取depmodeling/abacus-develop 仓库（github 和 gitee 二选一即可，国内建议用 gitee）

```shell
1 # github 仓库
2 \$ git clone -b develop https://github.com/depmodeling/abacus-develop.git
3
4 # gitee 仓库
5 \$ git clone -b develop https://gitee.com/deepmodeling/abacus-develop.git
```

编译 SIAB 模拟退火（SA）方法生成数值原子轨道的代码在 SIAB文件夹里，采用C++编写，目前Makefile仅支持intel编译器。

SIAB 程序有串行（makes）和并行（makep）两种版本，一般情况下串行版本即可满足使用需求。

3 \\$ make s

编译成功后，在目录 abacus-develop/tools/SIAB/SimulatedAnnealing/source 下可以看到生成数值原子轨道的可执行程序 SIA_s.exe，这里s代表编译串行版(serial)。此外，对于并行版本，SIAB 还可以结合MPI编译并行版，通过命令make p指令得到 SIA_p.exe 可执行程序，这里p代表paralel，但注意并行版只对某些特定算法进行并行，之后对代码进行一个更好整理之后我们会给出并行版的教程。

## 运行例子

进入tests_s 目录，可以看到H-6-0.6.15.dat文件，这是用来生成数值原子轨道所需要准备的输入文件，从ABACUS 的平面波自洽迭代计算后获得，现在该文件已经改名为OUT.\$suffix/orb_matrix.0.dat。

此外，还有一个输入文件INPUT如下：

1 <PW_QS>   
21 // bool: calculate the spillage. 1/0   
30 // bool: restart or not. 1/0   
41 // bool: output the file. 1/0   
51 // integer: number of structures.   
6 H-6-0.6.15.dat // Each structure's file name occupies one line   
7 </PW_QS>   
8   
9<PARALLEL>   
101 // integer: number of k points (for parallel version)   
11 1 // integer: number of pools (for parallel version)   
12 </PARALLEL>   
13   
14 <BLOCK_NE>   
1515 // can be ignored now   
16</BLOCK_NE>   
17   
18 <SCHEME_VALUE>   
19 2 // The way to calculate spillage, 2 means averaged spillage   
20<SCHEME_VALUE>   
21   
22 <METROPOLIS>   
23 1.0e-8 // real: start temperature for minimizing spillage   
24 0.5 // real: cooling rate for the temperature in minimizing spillage   
252 // integer: number of temperatures (spillage)   
26 500 // integer: number of steps per temperature (spillage)   
27   
28 1.0e-2 // real: starting temperature for the kinetic energy   
29 0.8 // real: cooling rate for temperature in minimizing kinetic energy   
301 // integer: number of temperatures (kinetic)   
31 100 // integer: number of steps per temperature (kinetic)   
32   
330.05 // real: delta kappa for kinetic energy   
3449 // integer: selectly output information every n steps   
35

36100 // integer: change accept rate every n steps，can be ignored now   
370.4 // real: acceptance rate (high)， can be ignored now   
380.2 // real: acceptance rate (low)，can be ignored now   
39   
4050 // real: max allowed kinetic energy (Rydberg)   
41 0.01 // real: 'dr' for the 1d r grid to compute kinetic energy   
42 1 // integer: 1: Kin 2: Ecut，can be ignored now   
43 </METROPOLIS>   
44   
45<BANDS>   
46 1 // bool: to control the number of bands   
471 // int, the start band index(>0).   
48 1 // int, the end band index(<provied bands).   
49 <BANDS>   
50   
51 <OPTIMIZE>   
52 2 // Number of levels.   
53 # label / na / skip / lmax / each L /   
54 01 2 new 1 1 1   
55 01 2 new 1 1 1   
56 01 2 new 1 1 1   
57 </OPTIMIZE>   
58   
59 <PLOT>   
600.01 //dr(a.u.) of uniform mesh. Attention!!dr will affect kinetic energy minmized largely.   
61-6 //xmin   
62 1 //zed， chosen as valence charge.   
63 0.01 //dx   
64 6.0 //xmax   
65 </PLOT>   
66   
67 <CAL_C4>   
680 // print out the C4, this section can be ignored now   
69 2   
70 ./FILE/Si-S.ORBITAL   
710   
72 ./FILE/Si-P.ORBITAL   
731   
74 </CAL_C4>   
75   
76 <TEST>   
770 // for tests, can be ignored now   
78 14.0 // rcut， only useful for test program   
79 0.01 // dr，for simpson integral   
80 2 // test eigenvalue index   
812 // 1max   
82 </TEST>

C4：代表4个重要参数：α,,l,nq，α是原子种类的指标，ε是角动量1对应的径向函数的个数，nq是对应的球贝塞尔函数的个数。

该输入文件包含 SIAB程序需要读取的一系列参数，基本上重要参数的说明都有列上去，一些可忽略的参数可以先不用设置，之后我们会推出更详细的关于此输入文件的说明文档。

准备好该输入文件后，可以通过如下命令执行SIAB程序：

1 \\$ \~/abacus-develop/tools/SIAB/Simulatedannealing/source/SIA_s.exe

运行完毕后，可以得到一系列输出文件，其中ORBITAL_1U.dat则是我们需要的数值原子轨道文件。我们可以通过文件的前几行获取关键信息：这是H原子的轨道文件，平面波截断能为100Ry，截断半径为6Bohr，最大角动量为1，角动量s和p的径向函数的个数都为2，即2条s轨道和6条p轨道，共8条轨道。为了体现这些关键信息，在使用轨道的时候我们建议重命名该文件为H_gga_100Ry_6au_2s2p.orb。

2 Element H   
3 Energy Cutoff(Ry) 100   
4 Radius Cutoff(a.u.) 6   
5 Lmax 1   
6 Number of Sorbital--> 2   
> Number of Porbital--> 2   
8   
9 SUMMARY EnD

## 3.7.3ABACUS中数值原子轨道的生成脚本

我们需要采用平面波计算不同键长的二聚体，并且存在不同层级的轨道，因此也可以通过一个脚本来实现自动化整个流程：abacus-develop/toos/SIAB/Generate_Orbital_AlInOne.sh。该脚本需要读入一个输入文件 OR-BITAL_INPUT如下:

1 #1.exe_dir   
2 #---   
3 ExE_mpi mpirun -np 4   
4   
5 ExE_pw /home/liuyu/github/abacus-develop/build/abacus   
6   
7 EXE_orbital /home/liuyu/github/abacus-develop/tools/SIAB/SimulatedAnnealing/source/SIA_s.exe   
8   
9 #1-1   
10 #( In this part, the direction of the two used exe is provided )   
11   
12 #2.electronic calculatation   
13 #--   
14   
15 targets 14_Si_60 # element   
16 ref_bands 46 # reference bands   
17 nbands 812 # num of bands for calculate   
18 Ecut 60 # cutoff energy (in Ry)   
19 Rcut 67 # cutoff radius (in a.u.)

20 Pseudo_dir /home/liuyu/github/abacus-develop/tests/PP_ORB   
21 Pseudo Si_ONCV_PBE-1.0.upf   
22 smearing_sigma 0.01 # energy range for gauss smearing (in Ry)   
23 #11   
24 #( In this part ， some parameters of calculating are given )   
25   
26 #3.structure information   
27 #11   
28 #the Bond Length list for each STRU   
29 BLSTRU1 1.8 2.0   
30 BLSTRU2 2.4   
31 # if skip the orbital optimization for each STRU (1:skip, O:not skip).   
32 SkipSTRU00   
33 # set the objective structure list to fit for orbital optimization   
34 ListSTRU dimer trimer   
35 #1   
36 #(In this part, it gives us the bond length of the reference system(in angstrom) )   
37   
38 #4.orbital calculatation   
39 #1-   
40   
41 maxL 2 # the max angular momentum   
42 Level 2 3 # num of levels to generate orbitals(<=5)   
43 #(num) (the max ang) (num of S) (num of P) (num of D) (num of F) (num of G)   
44 level1 111   
45 level2 2111   
46 level3 2111   
47 #1   
48 #( In this part， some information of orbital is given )   
49   
50 #5.Metropolis parameters (in most cases do not need to change)   
51 #1   
52 Start_tem_S 1.0e-3 # start temperature for optimize Spillage (default 1.0e-4)   
53 Start_tem_K 1.0e-1 # start temperature for optimize Kinetic (default 1.0e-2)   
54 Step_S 20 # num of steps for optimize Spillage (default 20)   
55 Step_K 10 # num of steps for optimize Kinetic (default 15)   
56 Delta_kappa O.01  # delta kappa (default 0.01)   
57 #1--1   
58 #( In this part ， some parameters of Metropolis is given. In most cases ，they   
59 #do not need to be changed ，only when you run into a situation ，that the   
60 #Kinnetic energy is larger than the maximum value allowed ， you can enlarge   
61 #the start temperature appropritely ， or you can enlarge the delta_kappa, e.g.   
62 #start_tem_k 1.0e-2 to 5.0e-1, delta_kappa 0.01 to 0.02. more steps can make   
63 #the orbitals better ，too)   
64 SUMMARYEND

该输入文件主要分为5个部分：

exe_dir

EXE_mpi:MPI并行计算命令，这里我们采用的是4核MPI并行进行dimer（或者trimer）的平面波计算。

EXE_pW：ABACUS可执行程序的绝对路径。

EXE_orbital：数值原子轨道生成程序 SIA_s.exe 的绝对路径。

## electronic calculatation

targets：可以任取，一般格式是\$原子序数_\$元素名_\$截断能。

ref_bands：参考体系的能带数，取值为电子总数的一半。注意这里有多个取值，原因是我们选择了多组参考构型，既有dimer，也有trimer。

nbands：平面波计算中的实际能带数，必须大于ref_bands，以保证包含非占据态。同理，由于多组参考构型，nbands 也有多个取值。

Ecut：平面波截断能，单位为Ry。

Rcut：数值原子轨道的截断半径，单位为Bohr。这里的多个取值表明，采用相同的一套参数，生成多个不同截断半径的数值原子轨道文件，根据第一部分的基础知识，截断半径会极大地影响数值原子轨道的质量，因此该参数可以帮助我们一次性生成多个不同截断半径的数值原子轨道文件。

Pseudo_dir：平面波计算采用的赝势文件所在文件夹的绝对路径。

Pseudo：平面波计算采用的赝势文件名。

smearing_sigma：平面波计算采用Gaussian smearing 的展宽，单位为Ry，一般取0.01 即可。

## structure information

BLSTRU1：第一组参考构型的键长，单位为埃。如上所示，第一组参考构型包含两个构型，键长分别为1.8和2.0埃。

BLSTRU2：第二组参考构型的键长，单位为埃。如上所示，第二组参考构型包含一个构型，键长为2.4埃。

以此类推，可以存在BLSTRU3、BLSTRU4等等。

SkipSTRU：是否跳过该组参考构型的优化，一般而言，都取为0即可。

ListSTRU：指定 BLSTRU1、BLSTRU2等等参考体系的具体构型，这里说明 BLSTRU1是dimer，BLSTRU2是trimer。如果取值缺省（第一个不可缺省)，则默认为dimer。

## orbital calculatation

maxL：最大角动量。

Levell：根据上文可知，轨道可以分为多个层级，这里level1是 SZ层级，最大角动量为1，角动量s和p的径向函数的个数为1和1，即1条s轨道和3条p轨道。

Level2：同上，这里level2的最大角动量为2，角动量s，p和d的径向函数的个数为1，1和1，即1条s轨道，3条p轨道和5条d轨道。level1和level2合在一起，是DZP层级。

Level3:同Level2。

Level:BLSTRU1 和 BLSTRU2 分别负责一部分轨道的生成，Level的2个取值对应的是 BLSTRU1和BLSTRU2的endlevel，即BLSTRU1用于level1和level2轨道的生成，BLSTRU2用于level3轨道的生成。

## Metropolis parameters

如注释所说，这部分参数实际上大多数情况下不需要改变。

Start_tem_S：优化 Spillage 的初始温度，更高的初始温度可以搜索更大的温度范围。

Start_tem_K：优化动能的初始温度，更高的初始温度可以搜索更大的温度范围。

Step_S：优化 Spillage 的步数。

Step_K：优化动能的步数。

Delta_kappa：模拟退火的惩罚函数κ所用的参数△。

准备好以上输入文件后，通过以下命令即可开始生成轨道：

1 \\$ \~/abacus-develop/tools/SIAB/Generate_Orbital_AllIn0ne.sh ORBITAL_INPUT

## 3.7.4ABACUS中数值原子轨道的测试流程

在生成轨道之后，为了保证基于数值原子轨道基组的第一性原理计算拥有与平面波基组相当的精度，我们需要针对该元素的一些常见的晶体结构做一些验证性计算。

以Al元素为例，我们可以选择 simple cubic（sc),body centercubic （bcc)，face center cubic （fcc）和 hexagonalclose-packed（hcp）这4种常见的晶体结构，采用平面波（Plane wave）基组和数值原子轨道基组计算这些晶体结构的平衡晶格常数、体弹性模量等基态性质，评估两种基组之间的误差随着截断半径、最大角动量以及轨道数量的收敛趋势，选择合适的数值原子轨道文件。例如以下就是用 ABACUS 的平面波和数值原子轨道DZP计算的铝的四种体相材料的体积-能量曲线，来自文献[1]。从图中可以看到，DZP轨道描述不同体相的铝的相对能量还是精度较高的。细心的读者会发现纵坐标给出的是绝对能量，因为两种基矢量采用了相同的赝势，但左图给出的能量更低，这是因为平面波更加完备，所以根据Honhenberg-Kohn 定理，体系总能量的密度泛函可以在更完备的基矢量下得到更好的收敛，即总能量更低。一般来说，平面波得到的总能量比DZP轨道小一些，大概在0.1-0.2eV/atom，是比较合理的。如果采用了同样的赝势，但两种基矢量给出的能量差别远远大于这个数值，则要检查一下是不是数值原子轨道产生出了问题。最后，大家有任何问题，欢迎写邮件到文档开头邮箱。

<!-- image-->  
Figure 8. Comparison of the total energies of Al sc,fcc, bcc, hcp structures as functions of volume per atom (a) using the plane wave basis (b) using the atomic DZP basis.

## 3.7.5参考文献

[1] M. Chen, G.-C.Guo,and L. He, Systematicaly Improvable Optimized Atomic Basis Sets for Ab Initio Calculations,J. Phys.: Condens. Matter 22,445501 (2010).

[2]M.Chen, G.-C.Guo,andL.He,Electronic Structure Interpolation via Atomic Orbitals,J.Phys.: Condens.Matter 23,325501 (2011).

## 3.8 数值原子轨道(三)：产生高精度数值原子轨道

## 3.8.1 PTG（PyTorch Gradient）方法

这篇文章是数值原子轨道系列的第三篇文档,除了第二篇文档提到的模拟退火算法之外,还可以使用Pytorch中的自动微分算法来最小化溢出函数。首先定义损失函数如下，并可以证明它与溢出函数是等价的：

$$
\Delta P S I \overset { \mathrm { d e f } } { = } \frac { 1 } { N _ { n } } \sum _ { n = 1 } ^ { N _ { m } } \| | \Psi _ { n } \rangle - | \tilde { \Psi } _ { n } \rangle \| ^ { 2 } ,
$$

其中

$$
\begin{array} { r } { | \tilde { \Psi } _ { n } \rangle \stackrel { \mathrm { d e f } } { = } \hat { P } | \Psi _ { n } \rangle , } \end{array}
$$

且 $\hat { P }$ 是由所有原子轨道张成的投影子，即 $\begin{array} { r } { \hat { P } = \sum _ { \mu v } | \phi _ { \mu } \rangle S _ { \mu v } ^ { - 1 } \langle \phi _ { v } | } \end{array}$ 。由 $\hat { P } ^ { 2 } = \hat { P }$ 及 $\langle \Psi _ { n } | \Psi _ { n } \rangle = \delta _ { m n }$ 可知，此处的损失函数 $\Delta P S I$ 与溢出函数是等价的。

## 3.8.2 PTG_dpsi （PyTorch Gradient with dpsi）方法

为了增加局域轨道做电子结构计算时的精度，损失函数的定义还可以被拓展，即在其中加入波函数的梯度，将总的损失函数定义为：

$$
\Delta P S I \overset { \mathrm { d e f } } { = } \frac { 1 } { N _ { n } } \sum _ { n = 1 } ^ { N _ { m } } [ \| \Psi _ { n } \rangle - | \tilde { \Psi } _ { n } \rangle \| ^ { 2 } + \| | \nabla \Psi _ { n } \rangle - | \nabla \tilde { \Psi } _ { n } \rangle | | ^ { 2 } ] .
$$

由于投影波函数 $\tilde { \Psi } _ { n }$ 是数值原子轨道的线性组合：

$$
| \tilde { \Psi } _ { n } \rangle = \sum _ { \mu } a _ { \mu n } | \phi _ { \mu } \rangle ,
$$

其中系数的表达式为

$$
a _ { \mu n } = \sum _ { \nu } S _ { \mu \nu } ^ { - 1 } \langle \phi _ { \nu } | \Psi _ { n } \rangle .
$$

那么投影波函数的梯度可以表示为数值原子轨道梯度的线性组合：

$$
| \nabla \tilde { \Psi } _ { n } \rangle = \sum _ { \mu } a _ { \mu n } | \nabla \phi _ { \mu } \rangle .
$$

因此，损失函数的梯度项可以转化为：

$$
\| | \nabla \Psi _ { n } \rangle - | \nabla \bar { \Psi } _ { n } \rangle | | ^ { 2 } = \langle \nabla \Psi _ { n } | \nabla \Psi _ { n } \rangle - \sum _ { \mu } a _ { \mu n } \langle \nabla \Psi _ { n } | \nabla \phi _ { \mu } \rangle - \sum _ { \nu } a _ { \nu n } ^ { * } \langle \nabla \phi _ { \nu } | \nabla \Psi _ { n } \rangle + \sum _ { \mu \nu } a _ { \mu n } ^ { * } a _ { \nu n } ^ { * } \langle \nabla \phi _ { \nu } | \nabla \phi _ { \mu } \rangle .
$$

根据经验，使用这个算法生成的双重-(+极化(DZP)基组，精度与模拟退火法或 PTG法生成的三重-(+双极化(TZDP)甚至四倍-(+三重极化(QZTP)基组相当。

在平面波计算中，除了数值原子轨道（二)：生成给定模守恒腰势的数值原子轨道提到的overlap 输出文件our.suffix/orb_matrix.o.dat之外，overlap 输出文件our.suffix/orb_matrix.1.dat 同样会作为生成轨道的输人文件之一，该文件保存与波函数梯度相关的如下overlap 项:

1．电子波函数梯度 $\nabla \Psi _ { i }$ 和局域轨道梯度 $\nabla \phi _ { \mu }$ 的overlap，也就是电子波函数梯度 $\nabla \Psi _ { i }$ 以及球贝塞尔函数梯度Vj(qr)之间的 $\mathsf { o v e r l a p } \langle \nabla \Psi _ { i } | \nabla j _ { l } ( q r ) \rangle$ ；

2.球贝塞尔函数梯度Vj(qr)之间的 $\operatorname { o v e r l a p } \langle \nabla j _ { l _ { 2 } } ( q _ { 2 } r ) | \nabla j _ { l _ { 1 } } ( q _ { 1 } r ) \rangle$

3．电子波函数梯度 $\nabla \Psi _ { i }$ 之间的 $\mathrm { { o v e r l a p } } \langle \nabla \Psi _ { i } | \nabla \Psi _ { i } \rangle$ 。

## 3.8.3 安装 Pytorch

PTG 和 PTG_dpsi方法采用 python 语言，无需编译，但是依赖 pytorch 包，下面介绍利用conda安装 pytorch的方法：

```shell
1 # 确定 conda 版本
2 \$ conda -V
3 conda 4.8.3
4
5 # 确定 python3 版本
6 \$ python3 -V
Python 3.5.2
8
9# 创建 python 环境
10 \$ conda create -n pytorch python=3.5
11
12 # 激活环境 (每次使用 pytorch 需要激活该环境)
13 \$ source activate pytorch
14
15 # 安装 pytorch
16 \$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
17
18 # 安装依赖库
19 \$ pip3 install --user scipy numpy
20 \$ pip3 install --user torch_optimizer
21
22 # 退出 python 环境（使用完毕后）
23 \$ source deactivate
```

## 3.8.4产生高精度数值原子轨道流程

首先下载 PTG_dpsi仓库

1 # github 仓库   
2 git clone -b main https://github.com/abacusmodeling/ABACuS-orbitals

接着准备输入文件SIAB_INPUT

```shell
1 #-1
2 #1．CMD & ENV
3 ExE_mpi mpirun -np 4
4 exe_pw /home/liuyu/github/abacus-develop/build/abacus
5 ExE_opt /home/liuyu/github/ABACUS-orbitals/SIAB/opt_orb_pytorch_dpsi/main.py
```

16 #3. Reference structure related parameters for PW calculation   
17 #For the built-in structure types (including 'dimer'，'trimer' and 'tetramer'):   
18 #STRU Name #STRU Type #nbands #MaxL  #nspin #Bond Length list   
19 STRU1 dimer 8 2 1 1.8 2.0 2.3 2.8 3.8   
20 STRU2 trimer 10 2 1 1.9 2.1 2.6  
该输入文件同样包含5个部分：

## CMD & ENV

EXE_mpi:MPI并行计算命令，这里我们采用的是4核MPI并行进行dimer（或者trimer）的平面波计算。

EXE_pW：ABACUS可执行程序的绝对路径。

EXE_opt：数值原子轨道生成程序PTG_dpsi的绝对路径。

## Electronic calculatation

element:元素名。

Ecut：平面波截断能，单位为Ry。

Rcut：数值原子轨道的截断半径，单位为Bohr。这里的多个取值表明，采用相同的一套参数，生成多个不同截断半径的数值原子轨道文件，根据第一部分的基础知识，截断半径会极大地影响数值原子轨道的质量，因此该参数可以帮助我们一次性生成多个不同截断半径的数值原子轨道文件。

Pseudo_dir：平面波计算采用的赝势文件所在文件夹的绝对路径。

Pseudo：平面波计算采用的赝势文件名。

sigma：平面波计算采用Gaussian smearing 的展宽，单位为Ry，一般取0.01即可。

## Reference structure related parameters for PW calculation

STRU Name：参考构型组别 STRU1、STRU2、STRU.

STRU Type：参考体系构型，一般可取dimer，trimer，tetramer。

nbands：平面波计算中的实际能带数，必须保证包含非占据态。

MaxL：最大角动量。

nspin：自旋量子数。

Bond Length list：键长取值列表，单位为埃。以STRU1为例，这一组共有5个不同键长的dimer作为生成轨道的参考构型。

## SIAB calculatation

max_steps：优化步数

LevelIndex：轨道分层的 index，取Levell、Level2、Level...

Ref STRUName：该层级的轨道采用哪一组参考构型生成。以上面的输入参数为例，Level1和Level2的轨道采用 STRU1组别的dimer 参考构型生成，Level3 的轨道采用 STRU2 组别的trimerr 参考构型生成。

Ref Bands：拟合参考体系的能带数，若输入“auto”则自动拟合所有基态。

InputOrb:是否在已有的轨道基础上生成新轨道，一般Levell设为none，即没有旧轨道;Level2一般以Level1生成的轨道为基础继续生成轨道，Level3往后同理。

OrbitalConf：轨道的具体配置，这里Level1是 SZ层级，Level2是DZP层级，Level3是TZDP 层级，实际上还可以继续设置更高层级的轨道。

## Save Orbitals

Index:取值 Savel、Save2、Saw.e..

LevelNum：将特定Level的轨道保存到一个轨道文件中。以上面的输入参数为例，分别将Level1、Level2、Level3 对应的轨道保存到单独的轨道文件中。

OrbitalType:即保存的轨道文件中,该轨道基组所属的层级。以上面的输入参数为例,分别为SZ、DZP、TDZP。

准备好输入参数文件SIAB_INPUT后，运行如下命令即可开始生成轨道：

1 \\$ python3 \~/github/ABACUS-orbitals/SIAB/SIAB.py SIAB_INPUT

程序正常结束后，轨道会分别保存在 Orbital_Si_SZ、Orbital_Si_DZP、Orbital_Si_TZDP中，并自动命名成标准格式（如Si_gga_6au_100Ry_2s2p1d.orb）。

## 3.8.5参考文献

[1]Peize Lin, Xinguo Ren and Lixin He, Strategy for constructing compact numerical atomic orbital basis sets by incorporating the gradients of reference wavefunctions,Phys.Rev. B 103,235131 (2021).

## 第4章基础功能

## 4.1 Kohn-Sham 密度泛函理论计算

## 4.1.1 平面波计算与收敛性测试

## 背景介绍

ABACUS(原子算筹)作为一款国产的电子结构软件，可用于对材料进行密度泛函理论（Density FunctionalTheory,简称DFT)的第一性原理（first-principles)计算,其主要核心功能之一是电子自洽迭代计算(SelfConsistentField，简称 SCF)，在给定材料微观层面的晶胞和原子位置的条件下，我们可以通过SCF 流程的计算获得电子结构的总能量、能级、电子波函数、电子密度等关键信息，这些信息可用于进一步计算得出材料体系的其它性质。然而，对于初学者来说，初次运行ABACUS难免不熟悉输入参数，容易出错，要熟练掌握好 SCF计算要领也颇具难度。因此，本文将主要针对初学者，详细介绍（1）如何采用 ABACUS 设置合理的输入参数完成一个 SCF 计算，以及（2）如何对体系做收敛性测试，以便确定所有计算参数。这里收敛性测试包括在平面波（Plane Wave，简称pw）基矢量下面对ecut（电子动能截断值）进行测试，以及对布里渊区的k点个数进行测试来确定这些参数。

对密度泛函理论计算不太了解的读者，我们想先介绍密度泛函理论的三个特点。第一，只要给定原子坐标和种类，密度泛函理论就可以用于预测该体系的许多物理和力学等性质。第二，对于不同的原子坐标排列方式以及不同元素种类，密度泛函理论软件提供了一系列参数用于调节计算的精度和效率，只有将参数设定在某一些正确的范围之内，得到的结果才有意义。反之，如果参数过于粗糙，则得到的结果可能也不收敛，因此不具有价值。第三，密度泛函理论计算是一种虽然较为流行的但也相对昂贵的科学计算方法。笔记本电脑只能跑得动几个原子的小体系计算，真正数十个原子乃至更大的体系需要借助超算进行长时间的计算（几个小时甚至几天)，而选择不同的参数可能会对最终运行时间有较大的影响。因此，基于以上三点认识，有必要更加细致地掌握密度泛函理论计算的具体参数，下面我们予以介绍。

## 输人文件简介

使用 ABACUS 中进行密度泛函理论或者第一性原理分子动力学计算时，一般需要 STRU、KPT、INPUT 三个最基本的文件作为ABACUS 的基本输人文件。STRU 文件提供元素、晶格、原子的基本结构信息；KPT 文件提供周期性边界条件下布里渊区采样的网格设置。INPUT文件主要提供计算所需的各种设置参数，在图一里我们分成了五部分且将在下面详细介绍。

## STRU和KPT文件简单介绍

## STRU文件

STRU 文件提供元素种类、原子质量（SCF中不会用到）、赝势文件（如果做密度泛函理论计算)、数值原子轨道（如果采用该基组做计算)、晶格矢量、原子位置、原子位置是否可移动等基本结构相关的信息。其中，晶格是晶体中原子有序排列的具体形式，我们可以选取一组晶格矢量来描述晶格。晶格确定后，原子的相对位置也可以通过选取不同坐标系（例如Cartesian 直角坐标系或者Direct晶格坐标系）确定下来。

例如，以下是一个金刚石硅的结构文件 (8个硅原子，周期性边界条件)

1 ATOMIC_SPECIES   
2 Si 28.085 Si_ONCV_PBE-1.0.upf   
3

<!-- image-->  
图4.1:电子自洽迭代计算流程。

```csv
4 NUMERICAL_ORBITAL
5 Si_gga_8au_60Ry_2s2pid.orb
6
7LATTICE_CONSTANT
81.8897261258369282
9
10 LATTICE_VECTORS
11 5.4307000000 0.0000000000 0.000000000
12 0.0000000000 5.4307000000 0.0000000000
13 0.0000000000 0.0000000000 5.4307000000
14
15 ATOMIC_POSITIONS
16 Direct
17 Si
18 0.0000000000
198
20 0.0000000000 0.0000000000 0.0000000000 1 1 1 mag 0.0
21 0.0000000000 0.5000000000 0.5000000000 1 1 1 mag 0.0
22 0.5000000000 0.0000000000 0.5000000000 1 1 1 mag 0.0
23 0.5000000000 0.5000000000 0.0000000000 1 1 1 mag 0.0
24 0.7500000000 0.7500000000 0.2500000000 1 1 1 mag 0.0
25 0.7500000000 0.2500000000 0.7500000000 1 1 1 mag 0.0
26 0.2500000000 0.7500000000 0.7500000000 1 1 1 mag 0.0
27 0.2500000000 0.2500000000 0.2500000000 1 1 1 mag 0.0
```

·第2行“元素原子质量赝势”中的“原子质量”在进行DFT计算时不可以省略不写，哪怕在SCF中不会用到。那么什么时候会用到呢？比如在进行分子动力学（Molecular Dynamics，简称MD，在 INPUT的calculation参数里设置）计算时，涉及运动的积分方程时用的是牛顿的第二定律F=ma，此时会使用原子的质量。另外注意，此行需要提供所使用的“赝势”文件名称。

·第5行提供LCAO基组所需的“某种元素的数值原子轨道”文件名称，而如果采用pw基组计算，不需要写第45行内容。

·第8行代表晶格整体缩放的一个长度，注意

$$
1 \ { \mathrm { A n g s t r o m } } = 1 . 8 8 9 7 2 6 1 2 5 8 3 6 9 2 8 2 \ { \mathrm { b o h r } }
$$

这里写1.8开头的小数意味着接下来的 LATTICE_VECTORS 部分可以写以 Angstrom 为单位的晶格。

·第1113行表示𝑥,y,z方向分别对应的三条晶格矢量。

·第16行“Direct”表示给出原子位置的方法是分数坐标（或者称为晶格坐标)，常用的还有 Cartesian(笛卡尔坐标方法)，就是直角坐标系下的坐标。

·第17行是元素种类的名称。

·第18行设置原子初始磁矩，如果INPUT 里的nspin 参数设为1，则不考虑磁性，这个参数不起作用。

·第19行表示体系的硅原子个数。

·第 20 27行给出所有8个硅原子的坐标；前三个数是坐标，之后的(111)三个数代表允许该原子在对应的x,y,z对应方向上移动，相反，(000)表示不允许在对应的方向上移动；同样，当考虑磁性计算的时候，“mag0.0”指定每个原子的初始磁矩，若设置此参数，则第18行的值将被覆盖。

关于赝势，还值得多提一些。平面波基组计算涉及采用描述电子和离子吸引作用的赝势，例如模守恒赝势或者超软赝势。需要注意的是，如果赝势换了，则相应的ecut能量截断值也需要重新测试收敛性。而基于局域原子轨道(Linear Combination of Atomic Orbitals，简称 LCAO)基组的计算则涉及赝势和数值原子轨道（NumericalAtomic Orbitals，简称 NAO)，需要使用者准备好相应文件。这里提到的赝势是一种描述核外电子和离子之间相互作用的近似方法，数值原子轨道是通过数值方法构建的描述电子波函数和电子密度等性质的基函数。需要注意的是，LCAO计算所采用的数值原子轨道要和对应的赝势匹配，即数值原子轨道是从给定赝势生成出来的，赝势如果改变了，相应的数值原子轨道也要重新生成，才能保证较好的精度。一般来说数值原子轨道的截断半径越长或轨道越多，则基矢量越完备，计算精度越高，但同时计算量也越大。

## KPT文件

KPT文件提供周期性边界条件下布里渊区k点采样的网格设置

1 K_POINTS   
0   
3 Gamma   
4kkk000

·第2行表示k点的总数，如果设置成“0”代表k点是自动生成的

·如果第二行是0，则第3行“Gamma”(T-centered Monkhorst-Pack method）是选择以 Gamma 点为中心的Monkhorst-Pack方法划分布里渊区网格，此外还可以使用“mp”方法，即最常用的Monkhorst-Pack方法。

·第4行的前三个整数代表网格沿着每个方向划分成几份，后三个数代表网格的平移量，000即代表不平移。

·计算立方形晶格时，k点各方向应取相同个数。在本次计算中，我们会将k的各方向从2取到8，测试不同k点下体系能量的收敛情况。

·在后文我们将介绍如何进行k点的收敛性测试，当选定一个k点后，若体系在某方向扩胞n次，该方向k点个数大致可以缩小到原来的 $\frac { 1 } { n }$ 。对于N原子以上的立方大体系，可通过在 INPUT 文件中设置参数 gamma_only取1，即只使用Gamma点计算，这个时候计算所需内存会显著下降，计算效率也会有提升。

·此外，k 点还有所谓的line mode，或者离散的k 点模式，若有兴趣的读者可以查看 KPT的介绍文档 https：//abacus.deepmodeling.com/en/latest/advanced/input_files/kpt.html

## INPUT文件关键输人参数

我们以金刚石结构硅原子体系（8个原子）作为示例，进行电子自洽迭代计算(SelfConsistent Field，SCF)。接下来让我们需要准备一个INPUT文件，我们把一些计算的关键参数分成如下5个部分。

## 基本参数

·suffix:suffix是用户可以自定义的后缀，运行ABACUS 可执行程序之后，输入文件所在的文件夹里会生成一个包含大部分运行信息的our.sufix 文件夹。例如，在这个例子里我们可以设为“Si”，运行后就会产生一个our.si文件夹。

·calculation:设置本次计算类型，例如本次文档主要展示“scf”（自洽电子结构计算)。scf(Self Consistent Field)、relax、cell-relax、md(MolecularDynamics)是较常用的四类计算。

1.scf(自洽电子结构计算)是一种用于求解电子基态电子密度的自洽迭代计算方法，通过送代来持续更新体系的电子密度，以达到收敛条件。通过电子自洽迭代，计算是否达到收敛由参数 scf_thr决定。

2.relax(结构驰豫计算)是通过调整体系中原子的位置来达到系统最稳定的状态的计算方法，每一次 relax 中都包含若干步的 scf 计算。在relax中，需要设置参数 force_thr_ev(eV/Angstrom)，代表所有原子中最大受力的收敛阈值。而relax计算的收敛需要每一步离子迭代时 scf 中的 scf_thr都满足，以及最后多步离子迭代后force_thr_ev 也满足。

3.cell-relax(结构驰豫计算)与relax 不同的是，其晶胞参数，比如晶格常数、晶格形状也会发生变化。除force_thr_ev外，还需要设置参数 stress_thr(kbar)，是晶胞感受到的应力的阈值。

4.md(分子动力学)基于牛顿力学原理，通过数值积分模拟粒子的运动轨迹，根据原子的势能计算相互作用力，因此势函数的选择比较关键，在ABACUS 里可以选择DFT来计算相互作用力，也可以选择DP势函数。

symmetry:是否考虑对称性，有如下三个选项

-1:不进行对称性分析

0:仅考虑时间反演对称性

1:进行对称性分析

如果打开对称性（设置为1)，布里渊区k点可以根据对称性进行简化处理，若体系有对称性，则可以减少所需计算的k点。因为每个k点都会进行一次 Kohn-Sham方程的求解，对称性分析后若k点减少则将提升计算效率。本次计算中开启对称性分析，设置“1”，关于对称性分析的测试还在进一步完善，如果计算结果奇怪，建议设置成“0”之后再进行计算，比较结果是否一致。

·pseudo_dir:计算中需要使用赝势来近似离子和电子相互作用势能，为计算提供赝势文件。pseudo_dir 指定STRU 文件中赝势文件所在的目录。本次计算将赝势和轨道文件都与INPUT、STRU、KPT文件放在一起，因此填人“.”，表示处在当前文件夹中。

ABACUS支持的赝势文件- -Si_ONCV_PBE-1.0.upf

“ONCV”代表模守恒赝势的种类，“PBE”是采用的交换关联泛函。

·orbital_dir:本次计算将采取pw（平面波轨道）基组（后文basis_type中设置)。若采用LCAO基组进行计算，则需要提供作为基组的轨道文件，在 STRU文件中指定轨道文件所在的目录。若选用pW 基组，则不需要轨道文件，不需要填写此参数。常用的轨道有：DZP（Double-{ valence orbitals plus SZ polarization，两条径向轨道和一条杂化轨道）、TZDP（Triple-( valence orbitals plus DZ polarization，三条径向轨道和两条杂化轨道)。

ABACUS支持的轨道文件 -Si_gga_7au_100Ry_2s2p1d.orb

“gga”代表GGA泛函；“7au”是轨道半径,半径越大，计算结果会更准确，但花费时间也更久；“100Ry”是推荐的ecutwfc值；“2s2pld”表明采用了DZP轨道，轨道数目越多，即基矢量更完备，计算也会更准确。

basis_type:计算的基组（指描述电子波函数的基函数)，在ABACUS 中常用的有两种：

pw:平面波基组（由于基矢量更完备，pw 计算将更准确，同时计算时间也更长)

LCAO:局域原子轨道基组(没有pw准确，但效率高)

·ecutwfc:平面波函数的能量截止（单位：Ry)，在平面波基组里是很重要的一个参数，其大小决定着作为基矢的平面波函数的个数多少，而基矢的多少，也决定了计算精度的高低。

在本次计算选用的LCAO 基组中，一般选取轨道文件上推荐的值，设置“100”Ry 即可。

本次计算中选择pw 基组。如前所述，pw 中不会用到轨道文件。

我们目前填写好的基本参数如下：

1 #Parameters (1.General)   
2 suffix Si   
3 calculation scf   
4 symmetry 1   
5 pseudo_dir ·   
6 orbital_dir ·   
basis_type pw   
8 ecutwfc 100

## SCF迭代参数

·scf_nmax:针对每个离子构型，scf电子迭代的最大次数为scf_nmax，可设为“50”或“100”次。半导体和绝缘体是较容易收敛的体系，一般20步scf以内即可达到收敛，对于金属体系或者费米面较为复杂的体系，有可能 50步仍未收敛。注意，如果做 relax,cell-relax,或者md 的时候发现有某些 scf 迭代次数达到最大时仍未达到收敛条件，建议先把计算停下来弄清楚原因后再继续算，有可能是因为离子构型或者k点等参数的设置不合理引起的。

scf_thr:代表两个相邻电子迭代步之间的电荷密度误差，也是SCF计算中判断是否收敛、完成计算的标准，对于LCAO，一般设置为“le-7”，可认为精度足够。对于pw，建议设置le-8或le-9的精度。

1 #Parameters (2. SCF iterations)   
2 scf_nmax 100   
3 scf_thr 1e-8

## 求解Kohn-Sham方程

·nbands:计算的Kohn-Sham轨道数目，在本次计算中无磁性，参数nspin取1（默认值)，程序目前采取0.5 ×max(1.2×occupied_bands,occupied_bands＋10)计算 nbands。对于Si，价电子数为4，每个能级填充可以填充自旋向上和向下2个电子，金刚石结构中共有8个原子，则

$$
n b a n d s = \operatorname* { m a x } ( 1 . 2 \times 8 \times 2 , 8 \times 2 + 1 0 ) = \operatorname* { m a x } ( 1 9 . 2 , 2 6 ) = 2 6
$$

·ks_solver:在不同基组中展开哈密顿矩阵的对角化方法，对于pw，可以选择cg (Conjugate Gradient，默认方法),bpcg (还不是太稳定、测试中)，dau(Davidson 算法)；对于LCAO，可以选择 genelpa(默认值)，scalapack_gux(Scalable Linear Algebra PACKage)。如果选用LCAO 基组，ks_solver 可设置为“genelpa”。

1 #Parameters (3. Solve KS equation)   
2 nbands 26   
3 ks_solver cg

---

## 第 141–160 页

## 展宽技术

·smearing_method:对于金属体系或者费米面附近较复杂的体系，电子自洽迭代方法往往不容易收敛，这个时候可以选择给电子提供一个光滑的占据函数，用来调节程序计算电荷密度和总电子数时候的电子占据函数，这对于费米面附近的电子态尤为重要。我们这里称为"smearing method”,或者称为展宽方法。具体来说,对于金属体系可以选择mp(methfessel-paxton)，mp2(2-nd methfessel-paxton)，也可以使用 gauss (也可以写作 gaussian)。因为我们计算金刚石 Si具有半导体性质，所以 smearing方法基本不起作用，默认可以选择gauss。此外，非导体计算可以选择fixed，半导体和非导体也可以选择fd (Fermi-Dirac)方法。

·smearing_sigma:给定展宽方法的能量范围（单位：Ry），默认是0.015Ry，我们按照通常情况给定“0.01”。

```python
1 #Parameters (4.Smearing)
2 smearing_method gauss
3 smearing_sigma 0.01
```

## 电荷密度混合

·mixing_type：进行新旧电荷密度混合时选用的方法，可选“plain”（简单电荷密度混合方法，如图1所示)、“pulay”、“broyden”，默认选择“broyden”算法。

·mixing_beta:新旧电荷密度混合时，新电荷的比例。这个参数越大，收敛越快，但不收敛的风险也会变大，不过取值大小并不会改变基态能量的结果。本计算中设置此参数为“0.7”。

·mixing_ggo:电子迭代过程中，可能出现混合电荷密度不收敛的情况，此时可以通过一种叫 Kerker预处理的方法来加快收敛。本计算关闭Kerker预处理，设置此参数为“0”。

1 #Parameters (5.Mixing)   
2 mixing_type broyden   
3 mixing_beta 0.7   
4 mixing_gg0 0

到此，我们完成了对主要计算参数的设置，此时集成了以上主要参数的INPUT文件如下：

1 #Parameters (1.General)   
2 suffix Si   
3 calculation scf   
4 symmetry 1   
5 pseudo_dir ·   
6 orbital_dir ·   
basis_type pw   
8 ecutwfc 100   
9   
10 #Parameters (2. SCF iterations)   
11 scf_nmax 100   
12 scf_thr 1e-8   
13   
14 #Parameters (3. Solve KS equation)   
15 nbands 26   
16 ks_solver cg   
17   
18 #Parameters (4.Smearing)

## 收敛性测试

## Ecut收敛 (只对平面波有效)

平面波（pw）作为一种可以用于描述周期性边界条件下的电子波函数和电荷密度基矢量，它们都是正交的,且可以通过一个ecutwfc 参数来控制基矢量的个数，ecut其代表了每一个平面波所对应的动能，如果ecut取得越大，则基矢量可以描述震荡得越剧烈的物理量（例如氧原子的2p轨道)，那么计算结果就会越精确，但同时所带来的机时成本消耗也越大。因此我们采用pW 基组做真正计算前，需要对ecut进行测试来获得一个足够准确且效率也高的取值。

保持前文用于计算的基组为“pw”，进行ecut的收敛性测试，ecut取值范围为：20\~100 Ry。这个例子在4进程下启动 MPI进行并行计算，一般推荐总核数(=线程数×进程数)取2的n次方或者n倍。一个较为粗糙、但基本不会造成计算资源浪费的选取n值原则是：体系有多少个原子，不要用超过这个原子个数太多的总核数进行并行计算。例如，体系如果有16个原子，不要用远大于16的总核数进行计算，一般取16的总核数或者更少就够了，具体需要测试。

1 mpirun -n 4 abacus

计算完成后，建议用脚本提取ecut数值下OUT.Si文件夹里running_scf.log文件中收敛的系统总能量，计算单原子能量并绘制其随着ecut的变化曲线。如图2，可以看到随 ecut增大，系统总能量趋于收敛。在 ecut=60Ry时，认为总能量收敛(与ecut=50Ry的能量差小于1meV/atom)。

<!-- image-->  
图4.2:体系里平均单个 Si原子能量(in eV/atom)随ecut (in Ry)变化。

## k点收敛

对具有周期性的体系进行计算时，DFT计算实际会对第一布里渊区中不同离散化k点的单独进行计算（例如每个k点都会单独求解一次依赖于该k点的Kohn-Sham方程)，并将计算结果进行积分。k点越多，则需要求的离散积分就越多，也将越近似连续积分的结果，但计算资源也会增加。因此，pW和LCAO都有必要对“需要多少k点”进行测试并得出合适的k点选取方案。

金刚石 Si的k点测试结果。同样在4进程下进行并行计算。

mpirun -n 4 abacus

计算完成后，提取各K点下OUT.Si文件夹里running_scf.log文件中收敛的系统总能量和计算时间，绘制这两个参数随K点的变化曲线。此处， $N _ { k }$ 表示各方向上的k点取值个数， $\mathrm { ^ { * } E n e r g y } - N _ { k }$ ”图中采用单原子的能量。

分析图3.3和图3.4可知，在 $N _ { k } = 6$ 时，结果已收敛，能量不会再发生较大的变化（与 $N _ { k } = 5$ 的能量差小于1meV/atom)，而此时花费时间132 秒，相比于更大的 $N _ { k }$ 点较低。因此选择6x6x6的k点可以同时获得准确的结果和高的计算效率。

<!-- image-->

(b)  
<!-- image-->  
图4.3:(a)体系里平均单个 Si原子能量(in eV/atom)随K点变化。(b)计算时间随K点变化。

提取出每个计算中实际使用到k点个数，如表一所示，实际使用k点数相比总k点数有较明显的简化。如前所述，INPUT文件中我们设置了参数 symmetry取“1”，因此在实际计算中利用了晶胞结构中的对称性，即倒易空间的积分没有使用所有的k点计算，而是巧妙的利用了对称性得到其它k点的信息，简化了布里渊区中的计算工作量。这也解释了图4中明明总k点数不同，计算时间却十分接近的原因。

表 4.1: $N _ { k }$ 点数，总k点数，实际使用k点数
<table><tr><td rowspan=1 colspan=1> $N _ { k }$ </td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>总k点个数</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>125</td><td rowspan=1 colspan=1>216</td><td rowspan=1 colspan=1>343</td><td rowspan=1 colspan=1>512</td></tr><tr><td rowspan=1 colspan=1>实际计算k点数 (开了对称性)</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>35</td></tr></table>

## 4.1.2ABACUS+LibRI做杂化泛函计算教程

## 一、介绍

杂化泛函（Hybrid Functional）是指在密度泛函理论框架中的交换关联项里面加入一部分的 Hartree Fock（简称 HF)的精确交换能。开源密度泛函理论软件 ABACUS 可以结合另一款国产开源软件LibRI软件进行杂化密度泛函计算，目前仅支持在数值原子轨道基组下使用该功能。可以通过dft_functional 参数指定所使用的杂化泛函类型，如可以选择hf (Hartree-Fock),pbe0(PBE0),hse(HSE06)以及 scan0杂化泛函。本教程以HSE杂化泛函为例，介绍如何在ABACUS 里调用LibRI做杂化泛函自洽迭代、求力和应力以及结构优化。

注1：使用ABACUS+LibRI做杂化泛函计算时，最大并行核数是 $N _ { a } ^ { 4 } N _ { k } ^ { 3 }$ ，其中 $N _ { a }$ 是原子个数， $N _ { k }$ 是k点个数。计算资源超出时可以运行，但会造成浪费。

注2：使用ABACUS+LibRI做杂化泛函计算时，因为内存消耗比较大，推荐给定计算资源的前提下，先尽量使用OpenMP多线程并行，再考虑使用MPI多进程并行。

## 二、杂化泛函的使用

## 1.ABACUS编译准备

如果要在 ABACUS 中使用杂化泛函进行计算，需要在编译ABACUS 的时候也编译Libxc、LibRI和LibComm三个软件包，具体请见线上文档 Advanced Installation Options ABACUS documentation。

注意在链接 LibRI、LibComm 时如果报错未定义的引用等，可以先注意检查 ABACUS 源代码下deps 文件夹下是否包含LibRI、LibComm两个文件夹。如果未包含这两个文件夹，或文件夹中无内容，在本地Github 仓库中可以尝试如下两条语句，以获取这两个子仓库内容：

1 git submodule init   
2 git submodule update --remote --recursive

## 2.采用杂化泛函进行电子自洽迭代计算

本教程在Gitee上准备了一个硅晶体使用杂化泛函做自洽计算（SCF）的例子（Gitee 的下载链接)，以下是INPUT文件及相关的参数。由dft_functional设置为hse可知，该例子使用的杂化泛函为HSE 泛函。KPT文件取的是444的布里渊区k点。

1 INPUT_PARAMETERS   
2 calculation scf   
3 basis_type lcao   
4 ntype 1   
5 nbands 8   
6 ecutwfc 60.0   
scf_nmax 100   
8   
9 dft_functional hse   
10 scf_thr 1e-7

杂化泛函相关的完整参数列表及解释见 Full List of INPUT Keywords /exact-exchange ABACUS documenta-tion。这里再进行简单概述：

## 。 泛函相关参数：

·exx_hybrid_alpha:杂化泛函中加入的HF精确交换能（Fock交换能）的比例α，即有 $E _ { x } = \alpha E _ { x } ^ { \mathrm { H F } } + ( 1 -$ $\alpha ) E _ { x } ^ { \mathrm { L D A / G G A } }$ 。如果df_1 functional设置为hf，则默认值为1。目前其他杂化泛函的默认值是0.25。但

是，如果是 SCANO泛函，有的文献取的是0.1，所以需要根据你想取的值进行设定[1]。

●exx_hse_omega:为HSE 泛函中的区间分割参数（range-separation parameter）ω，即有 $\begin{array} { r } { \frac { 1 } { r } = \frac { \operatorname { e r f c } ( \omega r ) } { r } + \frac { \operatorname { e r f } ( \omega r ) } { r } } \end{array}$ 默认值为 $0 . 1 1 \ \mathrm { ( b o h r ^ { - 1 } ) }$ ，此时为HSE06泛函[2]。

· $\mathtt { e x x \_ l a m b d a } \mathrm { : }$ 在 $\mathtt { b a s i s \_ t y p e }$ 设置为 $\mathtt { l c a o \_ i n \_ p w }$ 的情况下，用于补偿使用 $\mathtt { l c a o \_ i n \_ p w }$ 方法评估精确交换能时G=0处的发散点。默认值为0.3。

·exx_real_number:该参数设定为 True时，强制LibRI使用double数据类型，当设定为False时，强制LibRI使用complex 数据类型。当gamma_only=1时，默认为 True，gamma_only=O 时默认为False。

## 。 循环相关参数：

·exx_separate_loop:ABACUS 提供了两种迭代方法来评估精确交换能 $E _ { \mathrm { e x x } }$ 。当exx_separate_loop 设置为False 时：采用单层循环，即先进行GGA 循环，然后进行Hybrid循环，在该过程中，使用电子迭代来更新 $E _ { \mathrm { e x x } }$ 对应的哈密顿量 $H _ { \mathrm { e x x } }$ 。当exx_separate_loop 设置为True 时：采用双层循环，在内层循环中,进行自洽迭代并更新密度矩阵，在外层循环中，根据在内层循环中收敛的密度矩阵来计算 $H _ { \mathrm { e x x } }$ 。默认值为True，即采用双层循环计算。单层循环有利于难以自洽收敛的体系达到收敛，但会显著增加内存消耗。

. $\mathsf { e x x \_ h y b r i d \_ s t e p } \colon$ 在exx_separate_loop设置为True 的情况下,外层循环的最大迭代步数。默认值为100。

●exx_mixing_beta:在exx_separate_loop 设置为 True的情况下，内层循环每次迭代时，密度矩阵混合的mixing_beta取值，默认为1.0。

·exx_pca_threshold:为了加速四中心积分{ik jl>的计算，ABACUS 采用LRI方法，将原子轨道的乘积在辅助基函数(ABF)的基础上展开，即

$$
\Phi _ { i } \Phi _ { j } \approx \sum _ { a } C _ { i j } ^ { a } P _ { a }
$$

并利用PCA减小辅助基函数(ABF)的大小（即 $P _ { a }$ 个数)。阈值越大，ABF的数目越少，计算速度越快，计算精度越低。一个相对安全的值是1e-4，也是默认值。

·exx_ccp_rmesh_times：此参数决定计算 Column 势所需的截断半径比原子轨道的截断半径大多少倍。对于HSE 泛函，设置为1就足够了。但是对于PBEO，必须使用一个大得多的数字。当使用HSE泛函时，默认值为1.5，其他情况下默认值为5。

·张量筛选相关参数：针对杂化泛函计算过程中的物理量进行筛选可以加速计算。具体来说， $\mathsf { e x x \_ c } _ { - }$ threshold、exx_v_threshold、exx_dm_threshold、exx_c_grad_threshold、exx_V_grad_threshold分别是针对 $C _ { i j } ^ { a }$

$$
V _ { a b } = \langle P _ { a } | P _ { b } \rangle
$$

、密度矩阵、VCj、 $V V _ { a b } .$ 。阈值越大，筛掉的张量越多，计算速度越快，计算精度越低。具体请查看完整INPUT参数文档。

## 。 Cauchy-Schwartz不等式相关参数：

●exx_cauchy_threshold:在实际中，Fock 交换矩阵是稀疏的，利用Cauchy-Schwartz不等式，我们可以在进行显式求值之前找到每个矩阵元素的上界。小于exx_cauchy_threshold的值将被截断。阈值越大，筛掉的张量越多，计算速度越快，精度越低。一个相对安全的值是1e-7,也是默认值。不等式算法参见参考文献[3]。●exx_cauchy_force_threshold、exx_cauchy_stress_threshold与exx_cauchy_threshold类似,区别在于它们分别针对的是求力、应力计算中的Fock 交换矩阵元。

·opt_orb 相关参数：当dft_functional设置为opt_orb时使用，opt_orb参考文献[4]。本功能仅用于生成opt辅助基组，不进行杂化泛函计算。

·exx_opt_orb_Imax：球贝塞尔函数的最大角动量L值，opt-ABF 的径向部分用球贝塞尔函数的线性组合生成。

●exx_opt_orb_ecut：球贝塞尔函数展开的截断，在优化opt-ABF的时候采用的是球贝塞尔函数基组。

. exX_opt_orb_tolerance：解球贝塞尔函数零点时的阈值。

## 3.杂化泛函计算代价

杂化泛函的计算精度高，与此同时它的计算代价也比较高。在 ABACUS 的输入参数文件 INPUT 中，若exX_separate_loop 参数设为True（默认)，仅在SCF步骤中就涉及两层循环。每次内层循环完成，外层循环往前推进一步时，屏幕输出Updating EXXand rerun SCF。

一次 SCF需要的时间至少是以上两个循环涉及的单次电子迭代时间之和。对于单次电子迭代所需时间，在此提出一些已有的经验。以一步电子迭代的时间为衡量尺标，使用 DZP基组，CPU型号为Intel(R)Xeon(R) Gold6132CPU@ 2.60GHz，使用4核计算一个水分子为0.6s左右，使用14个核计算32个水分子为0.8s左右，使用14 个核计算64个水需要1.9s 左右。

若将 exX_separate_loop 参数设为False，即使用单层循环时，首先会进行GGA迭代直到自洽收敛，然后屏幕输出 Entering 2nd SCF,where EXX is updated，进行Hybrid 迭代，此时每进行一次电子步得到新的密度后，都会更新一次精确交换能。以一步电子迭代+更新精确交换能的时间为衡量尺标，使用DZP 基组，CPU 型号为Intel(R) Xeon(R)Gold 6132 CPU@ 2.60GHz，使用4核计算一个水分子为0.7s左右，使用14个核计算32个水分子为115s左右，使用14个核计算64个水需要330s左右。对于更大的体系，如2048个Si原子的晶体，使用DZP 基组，CPU型号为Intel(R) Xeon(R)Silver 4310 CPU@ 2.10GHz，用一个节点（56核）算时，PBE下一步电子迭代大概需要380s，而HSE一步电子迭代+更新精确交换能大概需要1680s。

在进行杂化泛函计算时推荐尽量使用多线程计算（OpenMP)，此时内存开销相对较小，计算速度相对较快。

## 三、使用杂化泛函做结构优化

## 1.数据准备

在Gitee 上我们准备了一个简单的使用杂化泛函做结构优化的例子。该例子是在LCAO 基组下，使用 HSE泛函，优化单个水分子的结构。文件夹中log.ref 是使用3.2.1版本的ABACUS软件包，v0.1.0版本的LibRI和LibComm 计算所得的屏幕输出。

## 2.输人文件

准备计算所需的INPUT文件、STRU文件、KPT文件，以及H、O原子对应的数值原子轨道文件。其中INPUT文件如下。注意该文件中指明了计算类型为relax，即不对晶胞做优化（cellrelax)，只对原子位置做优化(relax)。更多结构优化类型请看文档 Geometry Optimization ABACUS documentation。

1 INPUT_PARAMETERS   
2 calculation relax   
3 basis_type lcao   
4 ntype 2   
5 ecutwfc 60.000000   
6 scf_nmax 100   
gamma_only 1   
8   
9 dft_functional hse   
10 relax_nmax 100   
11 scf_thr 1e-6   
12 force_thr_ev 1e-2

在该例子中，结构优化包括多个离子步，每个离子步中都要做一次 SCF。由 INPUT 文件可知，SCF 收敛的标准由 scf_thr=le-6指定，或达到 SCF的最大步数 scf_nmax=100，并计算受力。根据上一个离子步计算得到的受力，计算下一个离子步的原子位置，计算收敛的标准此时为 force_thr_ev=le-2，或达到离子步的最大步数relax_nmaX=100。STRU文件如下，可见在结构弛豫步骤中，三个原子都可以移动。

ATOMIC_SPECIES   
2 0 16.00 0_ONCV_PBE-1.0.upf   
3 H 1.0O H_ONCV_PBE-1.0.upf   
4   
5LATTICE_CONSTANT   
61   
7   
8LATTICE_VECTORS   
92800   
10 0 280   
11 0028   
12   
13 ATOMIC_POSITIONS   
14 Direct   
15   
16 0 #label   
17 0 #magnetism   
18 1 #number of atoms   
19 0.677639488918 0.5227809096584 0.232500040128 m 1 11   
20   
21 H #label   
22 0 #magnetism   
23 2 #number of atoms   
24 0.641808457616 0.5785821615863 0.228644198512 m 111   
250.708889637644 0.5204300746076 0.175087721492 m111   
26   
27 NUMERICAL_ORBITAL   
28 0_gga_6au_60Ry_2s2p1d.orb   
29 H_gga_6au_60Ry_2s1p.orb

## 3.结果

结构弛豫（relax)后的原子结构可见OUT.ABACUS/STRU_ION_D。由输出文件可知，即使该例子中采用了相对稳定的构型，且scf_thr仅设为le-6，使用HSE做结构弛豫的计算代价仍然很高，使用6个核(Intel(R) Xeon(R)Gold 6132CPU@2.60GHz）计算需要5分钟左右。

## 四、参考文献

[1] Staroverov VN, Scuseria GE,Tao J,et al. Comparative assessment of a new nonempirical density functional: Molecules and hydrogen-bonded complexes[J].The Journal of chemical physics,2003,119(23):12129-12137.

[2] Aliaksandr V. Krukau, Oleg A. Vydrov, ArturF. Izmaylov, GustavoE. Scuseria; Influence of the exchange sreening parameteron the performance of screened hybrid functionals.J. Chem. Phys.14 December 2006;125 (22): 224106.

[3] Lin P,Ren X,He L.Eficient hybrid density functional calculations for large periodic systems using numerical atomic orbitals. Journal of Chemical Theory and Computation, 2021,17(1): 222-239.

[4]Lin P,Ren X,He L.Accuracyof localized resolution of the identity in periodic hybrid functional calculations with numerical atomic orbitals[J].The Journal of Physical Chemistry Letters,2020,11(8): 3082-3088.

## 4.1.3ABACUS收敛性问题解决手册

## 一、可选参数

mixing_type

描述：混合电荷密度所使用的方法

可选：broyden，pulay，plain

默认值：broyden

说明：没有特殊理由，不建议更改这个参数

mixing_ndim

描述：DIIS 混合计算混合系数时会保存的历史电荷密度数目

可选：int

默认值：8

说明：一般而言，这个值越大，收敛性越好，但是代价是内存消耗线性增加，8一般是够用的。

mixing_beta

描述：电荷密度的混合系数，1.0时表示全部用输出电荷，0.0时表示全部用输入电荷

可选：double

默认值：0.8 (nspin=1)；0.4 (nspin=2|4)

说明：一般而言，这个值越大，收敛越快，但是SCF跑飞不收敛的风险也越大；这个值越小，收敛越慢，但是收敛会更稳定，不收敛的情况会减少。

mixing_gg0

描述：电荷密度混合时所做的kerker预处理的强度。Kerker预处理会显著降低一些低频长程的电荷波动，

对于金属性体系的收敛帮助极大。

可选：double

默认值：1.0

说明：默认1.0代表ABACUS 默认打开kerker预处理，你可以手动设置成0.0将其关闭。一般而言,

绝缘体/分子/原子体系中kerker预处理的效果不明显，甚至有负优化，可以关闭。

mixing_gg0_min

描述：电荷密度的混合系数，所做的kerker预处理中，小于这一\$lqI\$的电荷分量不再进一步衰减

可选：double

默认值：0.1

说明：没有特殊理由，不建议更改这个参数

mixing_beta_mag

描述：磁密度的混合系数，只在nspin=2l4 时才会启用

可选：double

默认值：4\*mixing_beta，但是最大不超过1.6

说明：与电荷密度的混合类似，这个值越大，收敛越快，但是SCF跑飞不收敛的风险也越大;

这个值越小，收敛越慢，但是收敛会更稳定，不收敛的情况会减少。

mixing_gg0_mag

描述：磁密度的混合时所做的kerker预处理的强度

可选：double

默认值：0.0

说明：默认0.0代表ABACUS 默认关闭对磁密度的kerker预处理，你可以手动设置成非零将其打开。

一般而言，对于磁密度的Kerker预处理物理意义并不明确，不建议打开。

mixing_angle

描述：使用J.Phys.Soc.Jpn.82(2013)114706 中所描述的角度混合来更新电荷密度,

只在非共线计算，即nspin=4时才可以启用

可选：double

默认值：0.0

说明：默认0.0代表ABACUS 默认不使用这个方法，还是默认优先使用传统的DIIS 混合。

你可以手动设置成非零将其打开。这一方法可以解决非共线计算中磁矩收敛困难、收敛到错误磁态的问题。

## mixing_tau

描述：是否混合动能密度，只在metaGGA计算中才能使用

可选：bool

默认值：false

说明：ABACUS 默认不混合动能密度。但是打开一般会提升收敛性。

## mixing_dmr

描述：使用混合电荷密度的系数，同样混合密度矩阵。打开这一参数的前提必须同时设置mixing_restart

可选：bool

默认值：false

说明：ABACUS 默认不混合密度矩阵。如果你在进行需要密度矩阵构建算符的计算，

例如DFT+U/EXX/DeePKS，你可以试图打开mixing_dmr。经过我们大量的DFT+U算例测试,

mixing_dmr 配合一个合适的 mixing_restart会显著地提升 ABACUS 计算的收敛性。

## mixing_restart

描述：在第 mixing_restart步 SCF 清空mixing 的历史，直接使用上一步的输出作为这一步的输入，

一般与mixing_dmr 配合使用

可选：int

默认值：0

说明：ABACUS 默认不进行 restart，一个不适合的mixing_restart设置可能导致计算收敛不了。

## 二、如果收敛困难怎么办

## 非磁计算 (nspin=1)

非磁实际上可调的自由度不多，所有体系都可以遵循的原则如下：

1.尝试调小 mixing_beta

2.尝试调大mixing_ndim

分情况，你可以进一步有如下特殊的操作：

原子分子体系

原子分子的电荷高度局域，Kerker预处理一般都难以实际取得正优化，你可以选择关闭Kerkerk预处理，mixing_gg0=0.0。

## 半导体/绝缘体

虽然绝缘体部分电子态也很局域，但是Kerker预处理一般是安全的，即使负优化也不会很严重，但是你仍可以选择关闭Kerkerk预处理，mixing_gg0=0.0。

## 金属

调节不同的 mixing_ggO、mixing_ggO_min。

## metaGGA 计算

打开 mixing_tau=true。一般会解决 metaGGA 收敛困难，以及 EDIFF 慢于DRHO 的问题。

## 检查初始构型

一般而言，你不会在非磁计算遇到收敛性问题。如果真的遇到了，有很大概率初始构型太差，你可以调整构型，或者用先用宽松的收敛判据做几步弛豫计算，让体系构型更合理之后再调小scf_thr。

## 共线磁性计算(nspin=2)

磁性计算中所有体系都可以遵循的原则如下：

1.尝试调小 mixing_beta，并等比例调小 mixing_beta_mag

2.尝试调大mixing_ndim

分情况，你可以进一步有如下特殊的操作：

## 原子分子体系

孤立体系的磁性计算如果出现无法收敛，那大概率是出现一种极端的情况，即某一自旋全部来自某一轨道，不会叠加态。这种情况一旦出现，默认参数无法收敛。这时，请保证电荷密度和磁密度的混合系数完全一致：

1 mixing_beta 0.4   
2 mixing_beta_mag 0.4   
3 mixing_gg0 0.0   
4 mixing_gg0_mag 0.0

相信这个设置之后，问题就会解决。

## 半导体/绝缘体

与非磁建议一致。

## 金属

与非磁建议一致。

metaGGA 计算

打开 mixing_tau=true。一般会解决 metaGGA 收敛困难，以及 EDIFF 慢于DRHO 的问题。

## DFT+U计算

如果你的DFT+U计算无法收敛，我们推荐你做如下设置：

1 mixing_restart 10   
2 mixing_dmr 1

mixing_restart=10 代表 SCF 的计算会在第 10 步直接用第9步的输出电荷开启一个新的 SCF，在那之后 DFT+U的计算同时也会混合密度矩阵。在大多数情况下，10是一个安全有效的设置，但请务必记住 mix-ing_restart的设置是危险的，设置之后，你可能会遇到：

本来不设置这2个参数还可以收敛，只是收敛的不快，设置之后完全不能收敛了。

答：那是因为 mixing_restart的位置选取不合适，在 mixing_restart前一步的输出密度很差，用其做初始密度使得计算直接跑飞。这个时候，我们建议你观察不开mixing_restart时候，drho的收敛曲线，找一个距离基态近-些的位置做 mixing_restart（例如 drho<10^-3）。

本来不能收敛，设置这2个参数之后，仍然不能收敛

答：大概率是 mixing_restart的晚了，此时 SCF已经跑飞，结合drho的收敛曲线，适当提前 mixing_restart的位置。

## 弛豫计算

在弛豫计算中可以分多步做relax，会使用上一步的SCF的磁矩做下一步优化的初始磁矩。这样对于弛豫计算的收敛性帮助巨大。

## 非共线磁性计算(nspin=4)

在使用传统的DIIS 计算时，即mixing_angle=0.0，参数建议与共线磁性一致。但是如果尝试之后，仍难以收敛，或收敛到错误的磁态，那么我们建议你进行如下的设置：

1 mixing_angle  1.0

我们实现了J.Phys.Soc.Jpn.82(2013)114706中的新方法，这个方法寻找非共线基态的能力大大强于传统的 DIIS 混合。相信会对你有帮助。

## 三、其他FAQ

我可以在STRU中不设置初始磁矩，就设置成0吗？

答：可以。当 STRU 中所有元素的初始磁矩都是0的时候，我们会自动为体系中的所有原子都赋予大小为1a.u.的原子磁矩。但是我们仍建议你设置它，因为一个好的初始磁矩对于计算的收敛速度和正确性都有巨大的帮助。

我可以在计算中设置NUPDOWN来帮助收敛吗?

答：可以。一个合理的NUPDOWN设置一般会使得计算收敛地比较稳定。但是这么做的前提是，一定要对所做的体系很可靠的先验知识。

## 4.2分子动力学模拟

## 4.2.1ABACUS分子动力学使用教程

## 一、分子动力学方法简介

分子动力学（MolecularDynamics，简称MD)：一种模拟原子体系随时间运动的方法，被广泛的用来研究与原子运动路径相关的一些基本过程，如相变、扩散、化学反应等等。本教程旨在介绍ABACUS 中的分子动力学功能，分子动力学模拟重要的一件事情是如何选取精度和效率的平衡，其中精度高低主要取决于势函数的准确程度，而效率主要受限于求出每个原子的能量和受力的计算量。

经典分子动力学（ClassicaIMD，简称CMD)：通过构建描述原子间相互作用的势函数，获得每个原子的受力（受力等于能量对原子位置的导数)，再通过积分运动方程来获得每个原子的下一时刻位置，从而获得随时间演化的粒子位置和速度。当系统处在一定的密度、温度和压强等物理条件限制下，可以结合统计物理的方法计算物质的性质。具体来说，采取某个系综后，对粒子位置和速度采样，之后统计出体系的热力学宏观性质。

从头算分子动力学(ab initio molecular dynamics，简称AIMD)：也称为第一性原理的分子动力学方法（first-principles molecular dynamics，简称FPMD)，该方法采用的是第一性原理方法（例如密度泛函理论）来计算体系的势能面，因此计算量相比于经典分子动力学方法要昂贵很多。

ABACUS的分子动力学功能：支持第一性原理分子动力学FPMD方法，也支持经典的Lennard-Jones（LJ对势）的分子动力学模拟。此外，ABACUS 还支持深度势能分子动力学（Deep Potential Molecular Dynamics，简称DPMD）方法，此时需要编译DeePMD-kit软件包并在编译原子算筹软件时进行动态库的链接。

## 二、文档和算例下载地址

ABACUS里关于分子动力学方法的线上文档地址为：  
http://abacus.deepmodeling.com/en/latest/advanced/md.html  
ABACUS里关于分子动力学参数的线上文档地址为：  
http://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html#molecular-dynamics  
ABACUS的MD算例下载地址为（国内gitee）：  
https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/md  
可以采用的下载命令是：

1 git clone htps://gitee.com/mcresearch/abacus-user-guide.git

之后进人/abacus-user-guide/examples/md 目录或者采用Github 仓库地址：

注：算例仓库里面包含1_AIMD，2_LJMD（采用Lennard-Jones 经典势做 MD），3_DPMD（采用深度势能方法做MD）和4_FIRE（采用FIRE算法做结构优化）四个目录。

注：以上算例要在ABACUS v3.2.1版本及以上可以运行成功，并且我们强烈建议下载使用ABACUS 最新版本！

## 1.第一性原理分子动力学（AIMD)

进入1_AIMD目录，该目录提供了8个输入文件，使用的时候将 INPUT_\*复制为 INPUT即可（INPUT_\*代表从INPUT_0到INPUT_7共8个不同的INPUT文件，对应8种不同的MD算法)。该算例采用8原子的金刚石结构 Si，晶格结构放在 STRU 文件里，布里渊区的k 点采样放在KPT文件里。

8 个INPUT文件对应的MD方法如下：

INPUT_0：采用NVE 微正则系综

INPUT_1：采用NVT正则系综  
INPUT_2：采用Langevin热库  
INPUT_3：采用Anderson 热库，NVT系综INPUT_4：采用MSST方法  
INPUT_5：采用Berendsen 热库，NVT系综INPUT_6：采用 rescaling方法，NVT系综INPUT_7：采用rescale_v方法，NVT系综以INPUT_0为例：

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix Si_nve   
4 calculation md   
5 nbands 20   
6 symmetry 0   
7 pseudo_dir ../../../tests/PP_ORB   
8 orbital_dir ../../../tests/PP_ORB   
9   
10 #Parameters (2.Iteration)   
11 ecutwfc 30   
12 scf_thr 1e-5   
13 scf_nmax 100   
14   
15 #Parameters (3.Basis)   
16 basis_type lcao   
17 ks_solver genelpa   
18 gamma_only 1   
19   
20 #Parameters (4.Smearing)   
21 smearing_method gaussian   
22 smearing_sigma 0.001   
23   
24 #Parameters (5.Mixing)   
25 mixing_type pulay   
26 mixing_beta 0.3   
27 chg_extrap second-order   
28   
29 #Parameters (6.MD)   
30 md_type nve   
31 md_nstep 10   
32 md_dt 1   
33 md_tfirst 300  
这些参数在ABACUS 线上文档中均有说明，在这里再进行简单概述：  
·calculation：设置ABACUS计算类型，做分子动力学模拟请设置为 md。  
·esolver_type：给定原子位置进行系统能量的计算，默认Kohn-Sham密度泛函理论（ksdft)，还可以设置LJ势  
·chg_extrap：电荷外插法，在relax 和md计算中可用于优化电荷密度初始化，加速电子迭代收敛，MD中可以

设置为 second-order。

·md_type：MD 算法种类，默认是正则系统nvt，这里的例子选择的是nve，即NVE微正则系综。

。 md_nstep：MD 模拟的总步数。

·md_dt：MD 计算每一步的时间步长（单位是fs)，与md_nstep 共同决定MD 总时长。

。 md_tfirst：MD系统的初始温度（单位是K)。

## 2.Lennard-Jones 势函数

进入2_LJMD目录，ABACUS 软件包中提供了一个采用Lennard-Jones（LJ）经典势函数进行MD模拟的算例，该算例采用LJ对势作为能量求解器来做经典分子动力学，初始结构是边长20埃的正方体盒子中的 256个无序He原子。

## LJ对势

LJ对势的解析形式如下： $u ^ { i j }$ 表示原子i与原子j之间的相互作用能， $r ^ { i j }$ 表示这两个原子之间的距离，ε，σ和 $r _ { \mathrm { c u t } }$ 则是LJ势需要的3个参数。LJ势只考虑截断半径 $r _ { \mathrm { c u t } }$ 内的近邻原子j对中心原子i的能量贡献。

$$
u ^ { i j } = 4 \epsilon \left[ \left( \frac { \sigma } { r _ { i j } } \right) ^ { 1 2 } - \left( \frac { \sigma } { r _ { i j } } \right) ^ { 6 } \right]
$$

$$
F _ { i } = - \nabla _ { i } U = \sum _ { j } 4 \epsilon \left[ 1 2 \left( \frac { \sigma } { r _ { i j } } \right) ^ { 1 2 } - 6 \left( \frac { \sigma } { r _ { i j } } \right) ^ { 6 } \right] \frac { { \bf r } _ { i } - { \bf r } _ { j } } { r _ { i j } ^ { 2 } } H \left( r _ { \mathrm { c u t } } - r _ { i j } \right)
$$

$$
H ( x ) = { \left\{ \begin{array} { l l } { 0 } & { ( x < 0 ) } \\ { 1 } & { ( x > 0 ) } \end{array} \right. }
$$

## STRU

LJMD 是经典分子动力学，不需要提供 KPT，赝势和轨道文件，STRU的前面几部分如下：

1 ATOMIC_SPECIES   
2 He 4.0026   
3   
4 LATTICE_CONSTANT   
5 1.889727   
6   
7 LATTICE_VECTORS   
8 2000   
90200   
10 0020   
11   
12 ATOMIC_POSITIONS   
13 Cartesian   
14   
15 He #label   
16 0 #magnetism   
17 256 #number of atoms

ATOMIC_SPECIES：不需要提供赝势信息

NUMERICAL_ORBITAL：不需要添加轨道信息

## INPUT

INPUT文件中的参数也需要做少量修改：

INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4   
5 calculation md   
6 pseudo_dir ../../PP_ORB   
8 #Parameters (esolver)   
9 esolver_type lj   
10 1j_rcut 8.5   
11 lj_epsilon 0.01032   
12 1j_sigma 3.405   
13   
14 cal_force 1   
15 cal_stress 1   
16   
17 #Parameters (MD)   
18 md_nstep 5   
19 md_type nve   
20 md_dt 1   
21 md_tfirst 300   
22   
23 init_vel 1  
esolver_type：指定为lj就可以做Lennard-Jones 经典势函数。

lj_rcut:LJ对势的参数，即截断半径rcut。lj_epsilon：LJ对势的参数，即é。lj_sigma：LJ对势的参数，即o。init_vel：读取STRU中的原子速度信息，原子单位制。

## 3.深度势能 (Deep Potential)

ABACUS 软件包中提供了一个DPMD 的算例，进入3_DPMD 目录。该算例采用DP模型（DeePMD-kit产生的深度势能模型）作为能量求解器来做基于机器学习的分子动力学，初始结构是864个无序铝原子。

## DP模型

DeePMD-kit是一种基于机器学习的分子动力学模拟方法，该方法是通过使用第一性原理计算数据对深度神经网络模型进行训练，从而得到通用的多体势能模型 (DP模型)。

## 编译方式

ABACUS 采用 DP模型做MD计算需要编译与DeePMD-kit的接口，cmake以及 makefile 编译方式可以参考ABACUS 线上文档。

## INPUT文件设置

DPMD 是经典分子动力学，不需要提供KPT，赝势和轨道文件

INPUT文件中的参数需要做少量修改：

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4 calculation md

<table><tr><td>5</td><td></td></tr><tr><td>esolver_type</td><td>dp</td></tr><tr><td>7 pot_file</td><td>../../PP_ORB/Al-SCAN.pb</td></tr><tr><td>8 cal_force 9</td><td>1</td></tr><tr><td>10 cal_stress</td><td>1</td></tr><tr><td>11</td><td></td></tr><tr><td>12 md_nstep</td><td>3</td></tr><tr><td>13 md_type</td><td> msst</td></tr><tr><td>14 md_dt</td><td>1</td></tr><tr><td>15 md_tfirst</td><td>200</td></tr><tr><td>16 md_dumpfreq</td><td>1</td></tr><tr><td>17 md_restartfreq</td><td>1</td></tr><tr><td>18 19</td><td></td></tr><tr><td>msst_qmass 201 msst_vel</td><td>200 0.028</td></tr><tr><td>21 msst_vis</td><td>0.3</td></tr><tr><td>22</td><td></td></tr><tr><td>init_vel</td><td></td></tr><tr><td></td><td>1</td></tr></table>

esolver_type：设置为dp  
pot_file:DP模型文件路径  
md_dumpfreq：MD 输出文件MD_dump 中原子以及晶胞信息的输出频率  
md_restartfreq：结构文件 STRU_MD_\$istep 的输出频率，MD 续算文件Restart_md.dat的更新频率

## 三、MD输出和续算功能

## 1.MD 输出文件

ABACUS 的MD 模拟会产生3类输出文件：结构文件 STRU_MD_\$istep，续算文件Restart_md.dat，信息文件 MD_dump。

## 结构文件 STRU_MD_\$istep

参数 md_restartfreq控制 OUT.Ssufix/STRU/文件夹中结构文件 STRU_MD_Sistep的输出频率，该文件格式与输入结构文件 STRU 相同，主要用于MD 续算。这里istep 代表离子步，从0开始计数。

## 续算文件Restart_md.dat

参数 md_restartfreq 控制 OUT.\$suffix/Restart_md.dat 的更新频率，用于 MD 续算。

## 信息文件 MD_dump

参数md_dumpfreq 控制 OUT.\$sufix/MD_dump 文件的追加输出频率，该文件输出每 md_dumpfreq 步的 MD信息，包括MD步数，晶格常数，晶格矢量，晶格维里，原子编号，原子位置，原子受力，原子速度。其中，晶格维里，原子受力，原子速度可以通过输入参数dump_virial，dump_force，dump_vel控制是否输出。

1 MDSTEP : 0   
2 LATTICE_C0NSTANT: 5.397605400000 Angstrom   
3 LATTICE_VECTORS   
4 0.500000000000 0.500000000000 0.000000000000   
5 0.500000000000 0.000000000000 0.500000000000   
6 0.000000000000 0.500000000000 0.500000000000   
7VIRIAL (kbar)

54 0.000000000000 0.500000000000 0.500000000000

```csv
56 122.881221914392 1.206247195689 2.345023988024
57 1.206247195689 123.110079832453 -8.436042066928
58 2.345023988024 -8.436042066928 123.123976300784
59 INDEX LABEL POSITION (Angstrom) FORCE (eV/Angstrom) VELOCITY (Angstrom/fs)
60 0 Si 5.364727401370 2.714719749381 2.702800880457 0.310711926370 -0.086233730382
61 -0.043760493686-0.009222308089 0.005510922784 0.001122226311
62 1 Si 1.333700466782 1.360472064040 1.350800512757 -0.310711926370 0.086233730382
63 0.043760493686 0.009222308089 -0.005510922784 -0.001122226311
```

## 2 MD续算功能

如果 ABACUS 运行达到指定的MD步数，或者计算中断，可以通过MD续算功能继续之前的MD 计算。以上一个INPUT 文件为例：

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4 calculation md   
5 pseudo_dir ../../PP_ORB   
6   
7 esolver_type dp   
8 pot_file ../../P_ORB/A1-SCAN.pb   
9   
10 cal_force 1   
11 cal_stress 1   
12   
13 md_nstep 3   
14 md_type msst   
15 md_dt 1   
16 md_tfirst 200   
17 md_dumpfreq 1   
18 md_restartfreq 1   
19 md_restart 1   
20   
21 msst_qmass 200   
22 msst_vel 0.028   
23 msst_vis 0.3   
24   
25 init_vel 1

md_restart:控制续算的开关,在MD续算时将这个参数设为1,其他参数不变。当md_restart设为1,ABACUS会读取 \$read_file_dir/Restart_md.dat文件，从中获取当前 MD 步数istep 以及续算所需的其他参数如恒温器以及恒压器的信息，根据istep 从OUT.\$sufix/STRU/文件夹中读取相应的结构文件 STRU_MD_\$istep，之后就可以进行MD 续算了。

## 四、MD后处理

目前主流的可视化软件如VESTA、VMD、OVITO并不支持 ABACUS 的文件格式，因此我们需要通过一些后处理软件实现ABACUS 的 STRU、MD_dump 文件与常用文件格式的转化。我们比较推荐的后处理软件有：

## 1 ASE

官网：https://wiki.fysik.dtu.dk/ase/

用法参考 ABACUS 线上文档:http://abacus.deepmodeling.com/en/latest/advanced/interface/ase.html

## 2 dpdata

github 网址：https://github.com/deepmodeling/dpdata

gitee 网址：https://gitee.com/deepmodeling/dpdata

## 3案例

现在我们用LJMD算例来演示如何采用dpdata+OVITO的方式在MD计算完成后制作分子动力学轨迹动画。

## MD 计算

下载案例文件

1 \$ git clone https://gitee.com/mcresearch/abacus-user-guide

## 进入LJMD 文件夹

```perl
1 $ cd abacus-user-guide/examples/md/2_LJMD
```

修改INPUT参数，把md_nstep 改为100

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4   
5 calculation md   
6 pseudo_dir ../../PP_ORB   
8 #Parameters (esolver)   
9 esolver_type 1j   
10 lj_rcut 8.5   
11 1j_epsilon 0.01032   
12 lj_sigma 3.405   
13   
14 cal_force 1   
15 cal_stress 1   
16   
17 #Parameters (MD)   
18 nd_nstep 100   
19 md_type nve   
20 md_dt 1   
21 md_tfirst 300   
22   
23 init_vel 1

运行 abacus

1 \$ abacus

## dpdata转化格式

安装dpdata

```shell
1 $ git clone https://github.com/deepmodeling/dpdata.git dpdata
2 $ cd dpdata
3 $ pip install .
```

运行 dpdata，将 MD_dump 文件转化为GROMACS 的 gro 格式的文件 data.gro

```python
1 \$ python3
2 >>> import dpdata
3 >>> data = dpdata.LabeledSystem("./", fmt="abacus/md")
4 >>> data.to_gro("data.gro")
```

## OVITO制作动画

打开 OVITO，点击左上角 load file，选择上一步的data.gro 文件

<!-- image-->

右上角 rendering settings

---

## 第 161–180 页

<!-- image-->

点击 Render active viewport 即可保存动画文件

<!-- image-->

## 五、特殊的MD方法

## 1. Fast Inertial Relaxation Engine

ABACUS 中实现了一种特殊的基于MD的Fast Inertial Relaxation Engine (FIRE)结构优化方法，该算法比共轭梯度法的标准实现快得多，具有额外的速度修改和自适应时间步长，具有优异的优化效率。

以上面提供的4_FIRE 算例为例，我们可以介绍一下所需的参数如下：

INPUT_PARAMETERS   
#Parameters (1.General)   
3 suffix Si_fire   
4 calculation md   
5 nbands 6   
6 symmetry 0   
pseudo_dir ../../PP_ORB   
8 orbital_dir ../../PP_ORB   
9

10 #Parameters (2.Iteration)   
11 ecutwfc 30   
12 scf_thr 1e-8   
13 scf_nmax 100   
14   
15 #Parameters (3.Basis)   
16 basis_type pw   
17   
18 #Parameters (4.Smearing)   
19 smearing_method gaussian   
20 smearing_sigma 0.001   
21   
22 #Parameters (5.Mixing)   
23 mixing_type pulay   
24 mixing_beta 0.3   
25 chg_extrap second-order   
26   
27 #Parameters (6.MD)   
28 md_type fire   
29 md_nstep 100   
30 md_dt 1   
31 md_tfirst 300   
32   
33 force_thr_ev 0.001  
calculation：FIRE 算法是基于MD方法的，因此需要将calculation 设为 md  
md_type：采用FIRE算法，需要设为 fire  
md_tfirst:MD需要离子速度，因此需要通过设置初始温度来随机生成离子速度，或者在 STRU文件中手动设置初始速度  
force_thr_ev：离子受力的收敛阈值，当离子受力的每个分量均小于该阈值或者MD运行至最大步数时，MD终止

## 4.2.2ABACUS+DeePMD-kit做机器学习分子动力学

## 一、介绍

本教程旨在介绍结合ABACUS（中文名原子算筹，这里基于ABACUS3.8.0版本）和DeePMD-kit软件进行基于机器学习原子间势函数的分子动力学模拟。

DeePMD-kit是一款基于神经网络拟合第一性原理数据得到势能模型，用于分子动力学模拟的软件。无需人工干预，其可以端对端地将用户提供的数据在数个小时内转化为深度势能模型，可以保持量子力学精度准确性的基础上，将分子动力学的计算速度提升数个量级。

如果不熟悉 DeePMD-kit的读者，这里可以推荐大家先阅读 DeePMD-kit的相关原理文档和说明，熟悉这部分的读者可以跳过：

https://docs.deepmodeling.com/projects/deepmd/en/r2/

## 二、准备

## 1.案例下载地址

ABACUS的MD算例下载地址为（国内gitee）:   
https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/md   
可以采用的下载命令是：   
git clone https://gitee.com/mcresearch/abacus-user-guide.git   
之后进入/abacus-user-guide/examples/md/3_DPMD 目录   
或者采用Github 仓库地址：   
https://github.com/MCresearch/abacus-user-guide/tree/master/examples/md

## 2.安装相关软件

## ABACUS

这里默认读者已经准备好ABACUS 的安装环境和依赖软件，如还没有可以参考相关教程。

## DeePMD-kit

DeePMD-kit安装教程请参考DeePMD-kit的线上文档：  
https://docs.deepmodeling.com/projects/deepmd/en/r2/  
新手用户建议采用offlinepackage方式安装DeePMD-kit:  
https://docs.deepmodeling.com/projects/deepmd/en/r2/getting-started/install.html#install-off-line-packages  
如果用户希望直接使用DPA2做基于ABACUS 软件的分子动力学模拟，需要满足以下条件：  
安装DeePMD-kit v3 版本  
DeePMD-kitv3版本应该与采用的DPA2训练使用的DeePMD-kit版本一致  
DeePMD-kit v3 支持PyTorch backend

## ABACUS + DeePMD-kit

cmake安装方式：

在普通abacus 构建项目命令的基础上，指定DeePMD-kit的地址即可编译支持DP 势场的 ABACUS:

```shell
1 $ cmake -B build -DDeePMD_DIR=/dir_to_deepmd-kit ...
2 $ cmake --build build -j4
3 $ cmake --install build
```

如果采取的不是ofline packages 安装方式，则 TensorFlow 和PyTorch 的地址可能与 DeePMD-kit不同，则需要额外指定他们的地址：

```shell
1 $ cmake -B build -DDeePMD_DIR=/dir_to_deepmd-kit -DTensorFlow_DIR=/dir_to_tensorflow -DTorch_DIR=
2 /dir_to_pytorch ...
3 $ cmake --build build -j4
4 $ cmake --install build
```

TensorFlow 和 PyTorch 后端可能只有一个，也可能都有，用户应该根据自己的需求安装支持对应后端的DeePMD-kit。如果只有 TensorFlow 后端，则只需要提供 TensorFlow_DIR 即可，PyTorch 后端同理

make安装方式:

修改 Makefile.vars 文件，指定 DeePMD_DIR，LIBTORCH_DIR（可选）和 TensorFlow_DIR（可选）:

```python
1 DeePMD_DIR = /dir_to_deepmd-kit
2 LIBTORCH_DIR  = /dir_to_pytorch
3 TensorFlow_DIR = /dir_to_tensorflow
```

或者执行 make 命令时，指定 DeePMD_DIR，LIBTORCH_DIR（可选）和 TensorFlow_DIR（可选）：

```shell
1 $ make DeePMD_DIR=/dir_to_deepmd-kit LIBT0RCH_DIR=/dir_to_pytorch TensorFlow_DIR=
2 /dir_to_tensorflow
```

编译可能出现的问题:

采用offilne方式安装 DeePMD-kit，并使用 intel编译器编译ABACUS 时，可能会出现mkl库冲突，如

1 lib/libmkl_sequential.so.2: 对'mkl_lapack_sgesvda_batch_kernel' 未定义的引用

为避免影响原有的DeePMD-kit，可以新建目录/another_dir_to_deepmd，将原地址下的lib和 include文件夹复制至该目录下，然后删除 lib 中所有带mkl字符的库文件

```shell
1 $ cd /another_dir_to_deepmd/lib
2 $ rm *mkl*
```

## 三、机器学习分子动力学模拟

## INPUT文件设置

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4 calculation md   
5   
6 esolver_type dp   
7 pot_file ../../PP_ORB/A1-SCAN.pb   
8   
9 cal_force 1   
10 cal_stress 1   
11   
12 md_nstep 3   
13 md_type msst   
14 md_dt 1   
15 md_tfirst 200   
16 md_dumpfreq 1   
17 md_restartfreq 1   
18   
19 msst_qmass 200   
20 msst_vel 0.028   
21 msst_vis 0.3   
22   
23 init_vel 1

这些参数在ABACUS线上文档中均有说明，在这里再进行简单概述：

calculation：设置ABACUS 计算类型，做分子动力学模拟请设置为 md

esolver_type：给定原子位置后的能量求解方式，默认 Kohn-Sham 密度泛函理论（ksdft），还可以设置LJ势

(lj）或者深度势能（dp)，这里设置为dp

md_type：MD 算法种类，例如选择不同的系综 nve、nvt、npt 等

md_nstep：MD 模拟的总步数

md_dt：MD 计算每一步的时间步长（单位是fs)，与md_nstep 共同决定MD 总时长

md_tfirst:MD系统的初始温度(单位是K)

pot_file:DP 模型文件路径

md_dumpfreq：MD 输出文件MD_dump 中原子以及晶胞信息的输出频率

md_restartfreq:结构文件 STRUMD\$step 的输出频率，MD 续算文件Restart_md.dat 的更新频率

init_vel：从STRU文件中读取原子速度

## STRU文件

```csv
1 ATOMIC_SPECIES
2 Al 26.9815
3
4 LATTICE_CONSTANT
536.9005650344
6
7LATTICE_VECTORS
8 1 0 0 #latvec1
9 0 1 0 #latvec2
10 0 0 1 #latvec3
11
12 ATOMIC_POSITIONS
13 Cartesian
14
15 Al #label
16 0 #magnetism
17 864 #number of atoms
18 0 0 0 m 1 1 1 v 0.000116448153697 -3.54062829069e-05 6.37221822545e-05
19 0.0833333333321 0.0833333333321 0 m 1 1 1 v 6.31461715716e-05
20 -7.68881397243e-05 0.000126053061667
21 0.08333333333210 0.0833333333321 m 1 1 1 v -0.000203773691434
22 5.08681990052e-05 -5.02990480666e-05
2300.083333333331 0.0833333333321 m 1 1 1 v 0.00024862569861
24 -0.000253289379344 2.04134042994e-05
25000.1666666668 m 1 1 1 v 8.86312919936e-05 0.000160542135344 -0.000121839904824
26 0.083333333321 0.08333333321 0.16666666668 m 1 1 1 v 2.55030351768e-05
27 -3.68630188311e-05 2.89192521276e-05
28 0.083333321 0 0.25m 1 1 1 v -0.00135330676077 -3.26957881398e-053.15103467938e-07
2900.0833333333321 0.25 m 1 1 1 v 0.000131365976691 9.92067819339e-05 -0.00017808361164
30··
31 ··
32··
```

DPMD 除了不需要提供 KPT，赝势和轨道文件之外，其他设置与正常计算一致。

## 4.3采用ABACUS 进行表面计算（一)：静电势和功函数

## 4.3.1 背景

静电势（electrostatic potential)，也叫电势，是指将单位电荷从参考点移动到场内某一点所做的功。

在密度泛函理论的框架下，静电势 $V _ { s t a t i c }$ 定义为

$$
V _ { s t a t i c } = V _ { H } + V _ { e x t } + V _ { e f i e l d } + V _ { d i p o l e } + \cdot \cdot \cdot
$$

其中 $\begin{array} { r } { V _ { H } = \int \frac { \rho ( r ^ { \prime } ) } { | r - r ^ { \prime } | } d r ^ { \prime } } \end{array}$ 为电子产生的静电势， $V _ { e x t }$ 为离子势（赝势的局域部分), $V _ { f i e l d }$ 为外加电场项, $V _ { d i p o l e }$ 为偶极修正项，输出静电势和输出局域势的差别在于有没有输出交换关联势。

在处理表面等体系时，静电势可用于计算功函数等物理量，并进一步用于催化活性、化学反应机理以及电子输运性质等性质的计算与分析。

ABACUS 提供了直接输出静电势的接口，并提供了对其进行后处理的 python 脚本。

## 4.3.2ABACUS中计算静电势具体流程

此功能在平面波基组、原子轨道基组下的 Kohn-ShamDFT（KSDFT)，以及随机波函数密度泛函理论和无轨道密度泛函理论中均适用。接下来，我们以平面波基组下的 KSDFT为例介绍其使用方法，如果需要使用其它的能量求解器，只需要设置 INPUT 文件里的 esolver_type、basis_type 更换能量求解器并设置相应求解器的参数即可。

## 计算实例

## 输出静电势

下面是输入文件的示例：

INPUT文件记录计算所需主要参数，为了输出静电势，只需要设置out_pot2 即可（设置out_pot 1 时，会输出总的局域势，2相比于1 少输出了交换关联势，具体可以参考线上文档https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html#out-pot）。

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix example   
4 calculation scf   
5 ntype 1   
6 nbands 40   
7 symmetry 1   
8 pseudo_dir ·   
9   
10 #Parameters (2.Iteration)   
11 ecutwfc 60   
12 scf_nmax 100   
13   
14 #Parameters (3.Basis)   
15 basis_type pw   
16   
17 #Parameters (4.Smearing)

18 smearing_method gauss   
19 smearing_sigma 0.0074   
20   
21 #Parameters (5.Mixing)   
22 mixing_type pulay   
23 mixing_beta 0.7   
24   
25 out_pot 2

STRU 文件记录元素种类、质量、赝势，晶格矢量，原子坐标等信息

```csv
1 ATOMIC_SPECIES
2 Al 13 Al_ONCV_PBE-1.0.upf
3
4 LATTICE_CONSTANT
5 7.6513590200098225 // add lattice constant
6
7 LATTICE_VECTORS
8 0.707106781187 0.000000000000 0.000000000000
9 0.000000000000 0.707106781187 0.000000000000
10 0.000000000000 0.000000000000 5.000000000000
11
12 ATOMIC_POSITIONS
13 Direct
14
15 Al
16 0.0
175
18 0.000000000000 0.000000000000 0.000000000000 1 11
19 0.500000000000 0.500000000000 0.100000000001 11
20 0.000000000000 0.000000000000 0.200000000000 1 11
21 0.500000000000 0.500000000000 0.300000000000 11 1
22 0.000000000000 0.000000000000 0.400000000000 1 1 1
```

## KPT文件

```asm
1 K_POINTS
3 Gamma
4 20 201000
```

如上所示，为了输出静电势，只需要在INPUT中设置out_pot""2，计算完成后即可在OUTOUT.Ssuffx文件夹下找到下面的 ElecStaticPot.cube 文件。

ElecStaticPot.cube文件（记录了实空间格点上的静电势，单位是A.U.，下面的文档中，我们在括号中加入了注释。此文件可以直接在VESTA中可视化)

1 Cubefile created from ABACus SCF calculation. The inner loop is z index, folloved by y index, x index in turn.   
2 1 (nspin)

35（总原子数）0.00.00.0（原点坐标）  
4 27（沿 x 轴格点数 nx）0.200383 0.00000 0.00000 （晶格矢量 a1 的 1/nx，in Bohr。即晶格常数 a\*a1/nx)  
5 27（沿 y 轴格点数 ny）0.000000 0.200383 0.000000（晶格矢量 a2 的 1/ny，in Bohr)  
6192（沿 z 轴格点数 nz） 0.000000 0.000000 0.199254（晶格矢量 a3 的 1/nz，in Bohr)  
13（第一个原子的原子序数）11（赝势中的价电子数）0.0000000.00000.000000（该原子位置的笛卡尔坐标，in Bohr）  
13 11 2.705164 2.705164 3.825680  
9 13 11 0.000000 0.000000 7.651359  
10 13 11 2.705164 2.705164 11.477039  
1113 11 0.000000 0.000000 15.302718

```csv
12 (空间格点上的数据，z 坐标变化最快，接下来依次是y，x，一般为6 个数据一行，每走完z 的一次循环换一行)
13 -1.808618025e+01 -1.683249579e+01 -1.326226499e+01-8.726520963e+00 -4.859516723e+00 -2.648153424e+00
14 -1.754646057e+00 -1.082969997e+00 -7.712730497e-01 -5.301070017e-01 -1.865686936e-01 1.178808012e-03
15 1.531960521e-01 2.434659807e-01 3.236379649e-01 3.580121711e-01 3.871087692e-01 4.067752520e-01
16 4.191236489e-01 4.303464392e-01 4.269139054e-01 4.072584618e-01 4.012863417e-01 3.694437712e-01
17 3.419696568e-01 2.751686292e-01 1.986707660e-01 7.265002162e-02 -9.800789596e-02 -3.699329212e-01
18 ：
19 5.605648983e-01 5.596641131e-01 5.601221783e-01 5.558024312e-01 5.582286739e-01 5.509517334e-01
20 5.484144631e-01 5.474458329e-01 5.413456797e-01 5.310960172e-01 5.220139261e-01 5.057932112e-01
21 4.893954664e-01 4.661977108e-01 4.379332539e-01 3.906110446e-01 3.390039472e-01 2.575599458e-01
22 1.632711220e-01 1.092880266e-02 -1.657977733e-01-5.027618535e-01 -7.758727829e-01 -1.023213291e+00
23 -1.643162715e+00-2.406825325e+00 -4.139043775e+00 -7.343485448e+00 -1.119104878e+01 -1.435478802e+01
```

## 后处理

为了方便计算沿着z方向的平均静电势，我们在 tools/average_pot下提供了 python 后处理脚本 aveElecStat-Pot.py，在存放 ElecStaticPot.cube 的文件夹下运行

## python path_to_abacus/tools/average_pot

即可在此文件夹中得到存放了平均静电势的 ElecStaticPot_AVE 文件，如下所示。

ElecStaticPot_AVE文件（存放了沿z方向的平均静电势，单位是A.U.。此文件的前11行与ElecStaticPot.cube相同，之后的第一列数为沿着z轴的格点指标，第二列数为平均静电势)

```csv
1 Average electrostatic potential along z axis
2 1 (nspin)
3 50.00.00.0
4 27 0.200383 0.000000 0.000000
5 27 0.000000 0.200383 0.000000
6 192 0.000000 0.000000 0.199254
7 13 11 0.000000 0.000000 0.000000
8 13 11 2.705164 2.705164 3.825680
9 13 11 0.000000 0.000000 7.651359
10 13 11 2.705164 2.705164 11.477039
11 13 11 0.000000 0.000000 15.302718
12 iz average
130 -1.021155030e+00
141 -9.476023401e-01
152 -7.542571394e-01
163 -5.200099451e-01
1 7
```

<table><tr><td colspan="2">-5.039597208e-01</td></tr><tr><td>18 189 19 190</td><td></td></tr><tr><td></td><td>-7 .447469356e-01</td></tr><tr><td>20 191</td><td>-9.431350596e-01</td></tr></table>

可视化后如图所示：

<!-- image-->

## 功函数

功函数（work function)，又称逸出功，描述把电子从固体内部移到真空中所需的最小能量。

在 DFT计算中，它可以定义为 $\begin{array} { r l r } { V _ { v a c c u m } } & { { } - } & { E _ { F } } \end{array}$ ，其中 $V _ { v a c c u m }$ 为真空静电势，可以在ElectricPot_AVE 中读到, $E _ { F }$ 为体系费米能，可以用 grep EFERMI OUT.\$suffix/running_scf.log 命令从OUT.\$suffix/running_scf.log文件中提取出来。

对于上面的例子, $V _ { v a c c u m } = 0 . 5 6 3 \mathrm { R y }$ ，EF = 3.3293551883eV≈0.245Ry，因此功函数为 $V _ { v a c c u m } - E _ { F } =$ 0.318Ry。

如果大家使用有问题，欢迎写信联系 （见上）。

## 4.4采用ABACUS进行表面计算 (二)：偶极修正

## 4.4.1偶极修正介绍 (Dipole Correction)

偶极矩（Dipole Moment）是一个矢量，由正、负电荷中心距离和其所带电量乘积得到，方向规定为正电中心指向负电中心，采用原子建模表面时，往往表面会产生偶极矩。

ABACUS 在 xyz三个方向都采用周期性边界条件（Periodic Boundary Conditions），当使用密度泛函理论(Density Functional Theory，简称DFT）进行表面体系的电子结构计算时常采用超胞（Supercell）近似，这时所研究的体系在真空方向会出现两个表面。当两个表面不对称时（例如在某表面进行吸附原子计算)，这两个表面的电荷密度分布就会不同，这个时候两个表面的偶极密度不同且表面附近的静电势（Electrostatic Potential）会呈现不同的值。然而，周期性边界条件的存在会强制使得静电势也要满足周期性边界条件，这个时候周期性边界条件的存在相当于制造了一个非物理的人造电场（Artificial Electric Field)。

通常情况下，降低这个人造电场的方法是加大真空层的厚度，使得相邻的超胞之间没有相互作用，但这往往要大幅增加密度泛函理论的计算量。因此，为了消除周期性边界条件对表面偶极矩带来的影响，我们可以采用偶极修正（Dipole Correction）的方法，通过引入另外一个外电场来抵消这个人造电场，但注意引入的外电场需要存在于真空中（其位置是程序输入参数的一部分）才不至于影响所要研究的表面体系。经过修正后，人造电场将被抵消，真空电势则维持不变。

ABACUS 中实现了偶极修正功能[1]，可以有效的加速静电势收敛，避免超胞的真空取得太长，从而可以降低计算量。该修正功能主要体现在 Hartree项的处理上，并且存在相应的原子受力的修正，可以用于平面波(Plane-Wave）基组和数值原子轨道（Numerical Atomic Orbitals）基组计算，但注意该功能只能用于带真空层的表面体系。

## 4.4.2 文档与算例

ABACUS里关于偶极修正的线上文档地址为：

ABACUS 里关于偶极修正参数的线上文档地址为:https://bacus.deepmodeling.com/en/latest/advanced/input_files/input  
main.html#electric-field-and-dipole-correction

ABACUS 的偶极修正算例下载地址为（gitee）:

https://gitee.com/MCresearch/abacus-user-guide/tree/master/examples/dipole_correction/H2O

可以采用的下载命令是：

1 git clone https://gitee.com/MCresearch/abacus-user-guide

之后进入 /abacus-user-guide/examples/"dipole_correction/H2O/目录

或者采用Github仓库地址：

https://github.com/MCresearch/abacus-user-guide/tree/master/examples/dipole_correction/H2O

## 4.4.3偶极修正输人参数介绍

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix H20   
4 calculation scf   
5 nbands 6   
6 symmetry 0   
pseudo_dir ../../PP_ORB   
8   
9 #Parameters (2.Iteration)   
10 ecutwfc 100   
11 scf_thr 1e-6   
12 scf_nmax 100   
13   
14 #Parameters (3.Basis)   
15 basis_type pw   
16   
17 #Parameters (4.Smearing)

18 smearing_method fixed   
19   
20 #Parameters (5.Mixing)   
21 mixing_type pulay   
22 mixing_beta 0.7   
23   
24 #Parameters (6.Dipole correction)   
25 efield_flag 1   
26 dip_cor_flag 1   
27 efield_pos_max 0.0   
28 efield_pos_dec 0.1   
29 efield_amp 0.00   
30   
31 #Parameters (7.0utput)   
32 out_pot 2  
以算例提供的INPUT文件为例，使用偶极修正方法只需添加如下参数即可：

efield_flag:类型是 Bool,指定外加电场的开(1)或者关(0),默认值为0,注意应用偶极修正功能时,efield_flag必须设为true

dip_cor_flag：类型是Bool，指定偶极修正的开(1)或者关(O)，默认值为0

efield_dir：类型是Int，外加电场方向，可取0，1，2分别表示x，y，z方向，默认值为2

efield_pos_max：类型是Real，锯齿状电势最大值所在分数坐标，范围[0,1)，默认值为0.5

efield_pos_dec：类型是Real，锯齿状电势从最大值到最小值的长度（同样以分数坐标为度量)，默认值为0.1,范围[0,1)。以图1为例，0.5是最大值，0.6是最小值，那么0.50.6的区域必须放置在没有原子的真空层中

<!-- image-->  
图1.锯齿状势场分布图

efield_amp：类型是Real，电场强度，默认值为0，单位为a.u.，其中1a.u.=51.4220632\*1010V/m。在偶极修正时需要设为0，不为0则说明同时添加了外加电场和偶极修正

## 4.4.4预期结果

如图2所示，一个水分子位于超胞（尺寸为 $\mathrm { 3 \times 3 \times 1 8 A ^ { 3 } ) }$ 的中部（即位于9A)，那么按照上一部分的说明，锯齿应该放在没有原子的真空处，因此我们将电势的最大值所在的位置设为efield_pos_max设为0.0，之后设置电势从最大值到最小值的长度，即将 efield_pos_dec 设为0.1（在这个例子是1.8A)，如此锯齿即位于0.0～0.1的区域。

为了验证结果的正确性，我们可以画出静电势沿z轴(c方向）的变化如图3所示。在水分子上方（9-18A)和下方（0-9A）的真空层中，静电势在很长一段区间保持常数不变，出现了两个平台，这说明我们的偶极修正功能使用无误。静电势计算和作图方法请参考静电势和功函数计算文档。

<!-- image-->  
图2.一个水分子位于超胞中，红色代表氧原子，白色代表氢原子

<!-- image-->  
图3.静电势沿超胞Z轴变化图

## 4.4.5参考文献

[1] Phys.Rev. B 59,12301 (1999), https://doi.org/10.1103/PhysRevB.59.12301

## 4.5采用ABACUS 进行表面计算（三)：表面能计算

## 4.5.1 背景

将某一固体沿着某一晶面方向切开会形成相应的表面，许多化学反应都是依托在一些特定的表面系统上展开的。因此，计算表面能也是第一性原理计算经常会遇见的。

表面能（surface energy）定义为单位面积内表面体系与相应的体材料（bulk systems）能量之差，或者建立某特定表面需要做的功。不同表面的原子排布会对应不同大小的表面能，从能量的观点来看，表面能越低的表面体系越稳定，在现实世界里就越有可能稳定存在。

另外，需要注意的是，当沿着固体的某一晶面切开形成表面时，这个表面不一定是稳定结构。由于表面原子所经历的周围环境相比于体材料有所变化，所以往往表面原子会重新排布，只是排布的幅度有大有小。所以，-般来说，需要对表面进行结构优化计算，达到能量最低的稳定态。有一些常见的表面重构，已经被实验证实，例如硅的表面重构。

密度泛函理论（Density Functional Theory，简称DFT）也常用来计算材料体系的表面能，但如果使用的是周期性边界条件，往往构造的原子体系会有两个表面，以及一段真空。注意真空的长度一般不能太小（一般大于10埃)，要长到能量达到收敛。另外，如果表面是有极化的，可以采用dipole correction功能进行修正，这样真空往往可以取短一些。另外一方面，构成表面体系的原子层数一般来讲不少于3-5层，有时候需要检验更多层以达到收敛，达到收敛时体系最靠近中间的层一般和体材料的原子性质较为接近，这个可以作为表面厚度是否足够的判据之一，另外一个判据就是随着层数的增加，所得的表面能的值是否已经收敛。

用DFT计算材料体系的表面能时，需要进行两次自洽迭代（self-consistent field，简称 scf）计算，一次用于计算体材料体系中每原子的能量 $E _ { 0 }$ ，一次用于计算表面体系的总能量 $E _ { s l a b }$ ，最终表面能o定义为

$$
\sigma = \frac { E _ { s l a b } - N E _ { 0 } } { 2 A }
$$

其中 N 为表面体系包含的原子数，A为表面面积，分母上的2来自于如果计算中采用了周期性边界条件，即表面体系实际有上下两个表面。计算表面能时，需要计算 $E _ { 0 }$ 和 $E _ { s l a b }$ 两个能量。ABACUS 软件中，我们可以采用平面波基组、原子轨道基组下的Kohn-ShamDFT（KSDFT)，以及随机波函数密度泛函理论和无轨道密度泛函理论来计算它们。

## 4.5.2ABACUS计算表面能具体流程

接下来，我们以平面波基组下的KSDFT为例介绍其计算方法，如果需要使用其它的能量求解器，只需要设置INPUT文件里的esolver_type、basis_type 更换能量求解器并设置相应求解器的参数即可。

## 以下是几个构造表面结构的注意点：

由于 ABACUS在实空间的并行策略是将格点沿z轴分层，关于并行我们有两个建议：第一，为了使得 MPI并行时负载平衡，建议沿x方向设置真空，即让表面的法线沿着x轴，避免有些核分不到原子，影响计算效率;第二，如果并行的核数较多，建议沿z方向设置真空，即让表面的法线沿着z轴，因为每个核至少要分到一些z方向的格点，选取长边会使得可以并行的核数更多。

真空层厚度一般设置在10埃以上，具体需要调整真空层厚度做收敛性测试;

表面结构（slab）需要用内部的原子模拟体材料，因此其厚度也要做收敛性测试。

## 铝面心立方晶体(100)表面能计算

算例链接:htps://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Al_fcc100

## 计算体材料能量

这一步用晶格驰豫方法找到面心立方晶体fccAl的平衡构型以及平衡能量。

链接：https:/gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Al_fcc100/0_bulk运行完算例后，用 grep FINAL_ETOT_IS OUT.example/running_cell-relax.log 得到总能量为-1883.225eV，由于原胞内只有一个原子，因此 $E _ { 0 } = - 1 8 8 3 . 2 2 5 \mathrm { e V }$ 。

从OUT.example/STRU_ION_D文件中可以读到平衡晶格常数为 $a _ { 0 } = 7 . 5 8 6 \mathrm { B o h r }$ ，它将用于表面构型的构造。  
注意,如果是带有磁性的原子,计算体材料的时候也要加上磁性的计算,例如在INPUT文件里设置nspin=2。

## 计算表面体系能量

这一步我们搭建表面体系并计算其总能量。

链接:https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Al_fcc100/1_surface_5layers https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Al_fcc10o/2_surface_7layers

实际计算时，需要对表面构型的原子层数做收敛性测试，作为例子，我们提供了5层和7层的算例，其构型如下所示：

<!-- image-->

计算总能量时，可以根据实际需要选择是否做结构优化，这里为了结果更准确，同时兼顾效率，我们固定了内部原子，对表面上的原子结构进行了优化。

对于5层构型，用 grep FINAL_ETOT_IS OUT.example/running_relax.log得到总能量为-9415.005 eV，因此 $E _ { s l a b } = - 9 4 1 5 . 0 0 5 \mathrm { e V }$ 。

对于7层构型，总能量 $E _ { s l a b } = - 1 3 1 8 1 . 3 7 2 \mathrm { e V }$ O

## 计算表面能

对于上述构型，表面面积 $A = ( a _ { 0 } / \sqrt { 2 } ) ^ { 2 } = 8 . 0 5 7 \mathring { \mathrm { A } } ^ { 2 } ,$

对于5层构型，表面构型原子数N＝5，因此表面能 $\sigma = { \frac { E _ { \mathrm { { s l a b } } } - N E _ { 0 } } { \gamma A } } = 0 . 0 7 0 { \mathrm { e V / \AA } } ^ { 3 } = 1 1 1 3 { \mathrm { m J / m } } ^ { 2 }$

对于7层构型，表面构型原子数N＝7，因此表面能 $\begin{array} { r } { \sigma = \frac { E _ { \mathrm { s l a b } } - N E _ { 0 } } { 2 A } = 0 . 0 7 5 \mathrm { e V / \AA } ^ { 2 } = 1 1 9 6 \mathrm { m J / m ^ { 2 } } } \end{array}$

可以看到7层的构型相比于5层算出的表面能还差别较大，所以需要增加层数一直到表面能收敛。

## 2.硅金刚石结构(100)表面能计算

这里金刚石结构我们又名为CD（CubidDiamond）结构

算例链接:htps://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Si_CD100

## 计算体材料能量

这一步我们用晶格驰豫方法找到CD Si的平衡构型以及平衡能量。

链接：htps://gitee.com/mcresearch/abacus-user-guide/tre/master/examples/surface_energy/Si_CD100/0_bulk运行完算例后，用命令 grep FINAL_ETOT_IS OUT.example/running_cell-relax.log得到体系总能量为-214.516 eV，由于原胞内有两个原子，因此Eo=－107.258eV。

从 OUT.example/STRU_ION_D 文件中可以读到平衡晶格常数为 $a _ { 0 } = 1 0 . 3 5 1 \mathrm { B o h r }$ ，它将用于表面构型的构造。

## 计算表面体系能量

这一步我们搭建表面体系并计算其总能量。

链接：htps://itee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_energy/Si_CDlo0/1_surface我们搭建了一个9层的表面构型，如下图所示：

<!-- image-->

<!-- image-->

计算总能量时，可以根据实际需要选择是否做结构优化，这个例子里，为了计算省时，我们没有做结构优化。

用 grep FINAL_ETOT_IS OUT.example/running_scf.log 得到总能量为-961.337 eV，因此

$$
E _ { \mathrm { s l a b } } = - 9 6 1 . 3 3 7 \mathrm { e V } .
$$

## 计算表面能

对于上述构型，表面面积 $A = ( a _ { 0 } / \sqrt { 2 } ) ^ { 2 } = 1 5 . 0 1 5 \mathring { \mathrm { A } } ^ { 2 }$ ，表面构型原子数N＝9，因此表面能

$$
\sigma = { \frac { E _ { \mathrm { { s l a b } } } - N E _ { 0 } } { 2 A } } = 0 . 1 3 3 { \mathrm { e V } } / { \mathring { \mathrm { A } } } ^ { 2 } = 2 1 2 6 { \mathrm { m J } } / { \mathrm { m } } ^ { 2 } .
$$

与实验值 $2 1 3 1 ~ \mathrm { \ m J / m ^ { 2 } }$ [1]非常接近，但略大于另一个实验值 $1 3 6 1 ~ \mathrm { \ m J / m ^ { 2 } }$ [2]。

注：以上为表面能计算示例，若有问题欢迎邮件联系（见上)。

## 4.5.3 参考文献

[1]EagleshamDJ, White AE,FeldmanL C,et al.Equilibrium shape ofSi[J].Physical ReviewLetters,1993,70(11): 1643.

[2] Messmer C, Bilello JC. The surface energy of Si, GaAs,and GaP[J]. Journal of Applied Physics,1981,52(7): 4623-4629.

## 4.6采用ABACUS 进行表面计算 (四)：表面缺陷能和吸附能计算

## 4.6.1 背景

在研究表面的稳定性时，我们经常关注表面上形成空穴、或者表面上吸附其它分子的难易程度，这两种现象可以分别用表面缺陷能和表面吸附能描述。

表面缺陷能（surface vacancyenergy）定义为在表面上形成原子的空穴所需能量，缺陷能越低，则该表面越容易形成缺陷。

表面吸附能（surface adsorption energy）定义为原子、离子或分子（吸附物，adsorbate）附着在固体表面（吸附剂，adsorbent）的吸附过程中，二者结合时的能量下降，它描述了吸附剂对吸附物的吸附能力，吸附能越低，则吸附能力越强。

由于缺陷间或吸附物间的相互作用，上述两种物理量都与缺陷或吸附物在表面的覆盖率（coverage）θ有关,其定义为 $\theta = N _ { v / a } / N _ { \mathrm { s u r f a c e } }$ ，即缺陷数或吸附物数与表面原子数之比。覆盖率越低，则缺陷间或吸附物间相互作用越弱，计算时所需的超跑也越大，实际计算中，可根据需求选择合适的覆盖率。本文档中，为了节省计算量，采取较大的覆盖率θ=0.25。

下面我们分别介绍这两个物理量的计算方法，并展示如何使用ABACUS计算。

ABACUS 中，我们可以采用平面波基组、原子轨道基组下的Kohn-Sham DFT（KSDFT）和无轨道密度泛函理论等电子结构计算方法来计算所需的能量,也可以调用DepMD势函数。接下来,我们以平面波基组下的KSDFT为例介绍其计算方法，如果需要使用其它的能量求解器，只需要设置 INPUT文件里的 esolver_type、basis_type更换能量求解器并设置相应求解器的参数即可。

## 4.6.2ABACUS计算表面缺陷能具体流程

表面缺陷能 $E _ { v a c }$ 可定义为

$$
E _ { v a c } = E _ { s l a b / v a c } + E _ { b u l k } - E _ { s l a b }
$$

其中 $E _ { s l a b / v a c }$ 为带缺陷表面构型的总能量， $E _ { b u l k }$ 为体材料中每原子的能量, $E _ { s l a b }$ 为无缺陷表面构型的总能量，注意计算 $E _ { s l a b / v a c }$ 所需构型比 $E _ { s l a b }$ 少一个原子，即缺陷。

因此为了计算某种表面的缺陷能，我们需要进行三次自洽迭代（scf）计算，分别得到上述三个能量。

注意下面的例子中，为了提升效率，我们采用了较低的能量截断值（20Ry）以及较少的k点（gamma 点)，实际计算中，需要对二者做收敛性测试。

如果计算体系有磁性原子，要注意体材料的时候是否要加上磁性的计算。

## 计算实例：钼体心立方110表面缺陷能

链接:https:/gitee.com/mcresearch/abacus-user-guide/tree/master/examples/surface_vacancy_adsorption/0_vacancy/Mo

## 计算体材料能量

这一步我们用晶格驰豫方法找到体心立方（bcc）Mo的晶格常数以及平衡能量。

运行算例后，用 grep FINAL_ETOT_IS OUT.example/running_cell-relax.log 得到总能量为 -1861.827eV，由于原胞内只有一个原子，因此 $E _ { b u l k } = - 1 8 6 1 . 8 2 7 \mathrm { e V }$ 。

从 $0 \mathrm { U T . e x a m p l e / S T R U \mathrm { _ - } I O N \mathrm { _ - } D }$ 文件中可以读到平衡晶格常数为 $a _ { 0 } = 5 . 9 2 4 \mathrm { B o h r }$ ，它将用于表面构型的构造。

## 计算表面体系能量

这一步我们参考文献[1]，搭建了一个7层的Mo bcc(110)表面体系，每层包含4个原子，如下图所示：

<!-- image-->

计算其总能量时，可以根据实际需要选择是否做结构优化，这号

为了效率更高，我们没有做结构优化。

grep FINAL_ETOT_IS OUT.example/running_scf.log 得到总能量为”“-52129.48@ ev，因此Ejay =—52129.480 eV.

## 计算带缺陷的表面体系能量

第三步我们搭建带缺陷的表面体系，方法是从表面上移除一个原子，如下图所示，移除了表面中心的原子：

<!-- image-->

此算例同样没有做结构优化。

用grep FINAL_ETOT_IS OUT.example/running_scf.log得到总能量为-50265.933 eV,因此 $E _ { s l a b / v a c } =$ -50265.933 eV.

## 计算缺陷能

对于上述构型，最终的缺陷能 $E _ { v a c } = E _ { s l a b / v a c } + E _ { b u l k } - E _ { s l a b } = 1 . 7 2 0 \mathrm { e V } .$

## 4.6.3ABACUS计算表面吸附能具体流程

表面吸附能 $E _ { a d s }$ 可定义为

$$
E _ { a d s } = E _ { a d s o r b a t e / a d s o r b e n t } - E _ { a d s o r b e n t } - E _ { a d s o r b a t e } + \Delta Z P E
$$

其中 $E _ { a d s o r b a t e / a d s o r b e n t }$ 为吸附物与吸附剂结合后的总能量， $E _ { a d s o r b e n t }$ 为吸附剂体系的总能量， $E _ { a d s o r b a t e }$ 为吸附剂体系的总能量，△ZPE为零点能，注意计算 $E _ { a d s o r b a t e / a d s o r b e n t }$ 所需构型的原子数为 $E _ { a d s o r b e n t } \ $ $E _ { a d s o r b a t e }$ 之和。

因此为了计算吸附能，我们需要进行三次自治迭代（scf）计算，分别得到上述三个能量。

## 计算实例：Li原子在Mo bcc(110)表面的吸附能

链接:https://giteecom/mcresearch/abacus-user-guide/tree/master/examples/surface_vacancy_adsorption/1_adsorption/Li-Mobcc110

## 计算Li原子能量

这一步我们计算孤立Li原子的能量，方法是将一个Li原子放在一个足够大的原胞中，以排除Li原子间的相互作用。比如本例子中Li原子间距为20A。

用grep FINAL_ETOT_IS OUT.example/running_cel-relax.log 得到总能量为-183.156 eV，由于原胞内有两个原子，因此 $E _ { a d s o r b e n t } = - 1 8 3 . 1 5 6 \mathrm { e V }$ ·

## 计算Mo bcc(110)体系总能量

上文已经计算过 Eslab=-52129.480eV，因此这里 Eadsorbate=-52129.480 eV。

## 计算吸附体系总能量

这一步我们参考文献[1]搭建了以long bridge 形式吸附了Li原子的Mo bcc(110)表面体系，其中Li到 Mo表面的距离为2.229Å，如下图所示，绿色的原子为Li原子。

<!-- image-->

<!-- image-->

这里我们没有做结构优化，但在实际计算中，Li原子到Mo表面的距离应该通过结构优化得到。

用grepFINAL_ETOT_ISOUT.example/running_scf.log得到总能量为-52315.069eV，因此$E _ { a d s o r b a t e / a d s o r b e n t } = - 5 2 3 1 5 . 0 6 9 \mathrm { e V } .$

## 计算吸附能

对于上述构型，吸附能 $E _ { a d s } = E _ { a d s o r b a t e / a d s o r b e n t } - E _ { a d s o r b e n t } - E _ { a d s o r b a t e } = - 2 . 4 3 3 \mathrm { e V }$ .（此处忽略了零点能）

## 4.6.4参考文献

[1] Chen M, RoszellJ, Scoullos EV,et al.Effect of temperatureon the desorption of lithium from molybdenum (110)surfaces: Implications forfusionreactor first wallmaterials[J].The Journalof Physical ChemistryB,2O16,120(26): 6110-6119.

## 4.7采用ABACUS进行表面计算（五)：外加电场

## 4.7.1介绍

电场是电荷及变化磁场周围空间里存在的一种特殊物质。这种物质与通常的实物不同，它虽然不是由分子原子所构成的，但它却是客观存在的特殊物质，具有通常物质所具有的力和能量等客观属性。许多材料在外加电场作用下会表现出不同寻常的性质，例如：铁电体在外加电场的作用下会产生电极化，也可以通过施加的电场来控制磁性和自发极化，因此在信息储存和传输中有广泛应用；二维材料（如石墨烯）在外加电场下表现出不同的电子性质和可控制的物理性质，这种特性使其具有在电子学领域的广泛应用前景。在电子结构计算中，周期性边界条件下对体系施加电场较为容易，而施加磁场则困难很多。

因此，外加电场是密度泛函理论（Density Functional Theory，简称 DFT）软件中可以被实现且重要的功能。ABACUS 中实现了在表面体系中外加恒定电场的功能[1]。注意，在 ABACUS 中，外加电场功能存在相应的原子受力的修正，可以用于平面波（Plane-Wave）基组和数值原子轨道（Numerical Atomic Orbitals）基组计算，但注意该功能只能用于带真空层的表面体系。

本教程中将会展示如何在ABACUS 计算中使用加电场功能，为使得这个教程较为有趣，我们举一个半金属二维材料nanoribbon（见图1）加电场产生半金属的例子来讲解如何给体系添加电场，该DFT的计算工作曾发在2006 年的 Nature杂志上，具体可参考文献[2]。在这个教程里，我们使用的是数值原子轨道基矢量（NumericalAtomic Orbitals），碳和氢的轨道半径都是8a.u.，但实际上也可以使用平面波基矢量，只是前者效率更高所以在这里采用。我们还会使用到电子的自洽场（self-consistent field，简称 SCF）计算，非自洽计算（non-self-consistentfield，简称NSCF)，自旋非极化和极化计算等概念。在不考虑自旋时（即设置nspin=1时)，计算出的能带图中没有带隙，为金属；考虑自旋之后，带隙打开，成为半导体；在自旋极化基础上，外加恒定电场之后，其中一个自旋表现为半导体，另一个自旋则为金属。

<!-- image-->  
图1.Nanoribbon 结构图，黑框代表超胞大小，有真空。超胞里包含32个碳原子（棕色），超胞里接触真空的2个碳原子（每个表面一个碳原子）被2个氢原子（白色）饱和。

## 4.7.2软件和算例准备

ABACUS里关于外加电场的线上文档地址为：

http://abacus.deepmodeling.com/en/latest/advanced/scf/advanced.html#external-electric-field

ABACUS 里关于外加电场参数的线上文档地址为:https://bacus.deepmodeling.com/en/latest/advanced/input_files/inputmain.html#electric-field-and-dipole-correction

ABACUS的外加电场算例下载地址为（gitee）:

https://gitee.com/MCresearch/abacus-user-guide/tree/master/examples/electric_field/

可以采用的下载命令是：

1 git clone https://gitee.com/MCresearch/abacus-user-guide

下载后解压，之后进入/abacus-user-guide/examples/"electric_field目录

或者采用Github仓库地址：

https://github.com/MCresearch/abacus-user-guide

## 4.7.3采用加电场进行二维材料的能带计算

该算例中含有六个文件夹，解释如下：

1_unspin_scf：做非自旋极化（INPUT文件中的 nspin 设为1）的自洽场迭代 SCF 计算（INPUT文件中的calculation设为 scf)，输出自洽迭代之后的电子密度（将INPUT中的out_chg设为1)，注意二维材料的 KPT 文件，因为超胞大小是102.459512146645，所以对应的布里渊区k点取的是1101，因为实空间周期长度小的方向对应的布里渊区大，所以需要取更多k点得到收敛结果。

---

## 第 181–200 页

2_unspin_band：采用非自洽计算NSCF（INPUT文件中的calculation 设为nscf)，读入上一个步骤输出的电荷密度（INPUT文件中的init_chg设为 file)，从而可以读入体系的基态电荷密度进行能带计算。

3_spin_scf：同1_unspin_scf 的计算，只是把nspin 设为2。

4_spin_band:同2_unspin_band 的计算，只是把 nspin 设为2。

5_spin_elec_scf：同3_spin_scf的计算，只是又外加了电场的参数。

6_spin_elec_band：同4_spin_band的计算，只是又外加了电场的参数。

以下是算例5的输入文件 (包含电场参数)：

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix nanoribbon   
4 calculation scf   
5 nbands 70   
6 nspin 2   
7 pseudo_dir ../../PP_ORB   
8 orbital_dir ../../PP_ORB   
9   
10 #Parameters (Basis)   
11 basis_type lcao   
12   
13 #Parameters (Accuracy)   
14 ecutwfc 100   
15 scf_thr 1e-6   
16 scf_nmax 200   
17   
18 #Parameters (Smearing)   
19 smearing_method gaussian   
20 smearing_sigma 0.001   
21   
22 #Parameters (Charge mixing)   
23 mixing_type pulay   
24 mixing_beta 0.1   
25 mixing_ndim 20   
26 mixing_gg0 1.5   
27   
28 #Parameters (Efield)   
29 efield_flag 1   
30 efield_dir 2   
31 efield_amp 0.0019440124   
32 efield_pos_max 0.95   
33 efield_pos_dec 0.10

以算例提供的INPUT文件为例，使用外加电场功能只需添加如下参数即可：

efield_flag：类型是Bool，指定外加电场的开(1)或者关(O)，默认值为0

efield_dir：类型是Int，外加电场方向，可取0，1，2分别表示x，y，z方向，默认值为2

efield_pos_max：类型是Real，锯齿状电势最大值所在分数坐标，范围[0,1)，默认值为0.5

efield_pos_dec：类型是Real，锯齿状电势从最大值到最小值的长度（同样以分数坐标为度量)，默认值为0.1,范围[0,1)。以图2为例，0.5是最大值，0.6是最小值，那么0.50.6的区域必须放置在没有原子的真空层中，注

意这个例子设置的是0.95到0.10为电场区间。

<!-- image-->  
图2.锯齿状势场分布图

$$
1 \mathrm { a . u . } = 5 1 . 4 2 2 0 6 3 2 * 1 0 ^ { 1 } 0 V / m
$$

## 4.7.4预期计算结果

运行完算例2_unspin_band 后会通过非自洽计算得到体系非自旋极化能带图，横坐标是对二维材料布里渊区取k点，从Gamma点（坐标为000）取到π点（坐标为00.50)，取了100个k点算出来的能带图；纵坐标是能级，将费米面设置为0，取了费米面附近2eV的能量区间画出能带。以下两张图是同样取法。

运行完算例4_spin_band 后会得到非自洽计算得到的自旋极化能带图（图3)。

运行完算例6_spin_elec_band 后会得到自旋极化加电场的能带图（图4），注意这里所加的电场为0.1V/A,由输入文件 INPUT中的efield_amp 设置，但efield_amp 的单位是a.u.，需要换算。

<!-- image-->  
图3.采用PBE 交换关联泛函和非自旋极化得到的二维 nanoribbon 的能带图，可以看出费米面附近CBM（Con-duction Band Minimum）和VBM（Valence Band Maximum）重合，无带隙。

<!-- image-->  
图4.采用PBE交换关联泛函和自旋极化得到的二维nanoribbon的能带图。蓝色和红色代表自旋极化方向不同时对应的两副能带图像，可以看出不加电场时，两个自旋方向的能带图几乎一样，都有带隙。

<!-- image-->  
图5.采用PBE 交换关联泛函和自旋极化，再给体系加上0.1V/A的电场得到的二维nanoribbon 的能带图。蓝色和红色代表自旋极化方向不同时对应的两副能带图像，可以看出加了能带之后，其中一个自旋方向的能带图出现费米面附近的交叠，呈现金属性质，另外一个自旋方向的能带图依旧保持在费米面处的能隙。

## 4.7.5参考文献

[1] Phys. Rev. B 59,12301 (1999),htps://doi.org/10.1103/PhysRevB.59.12301

[2] Nature 444, 347-349 (2006), https://doi.org/10.1038/nature05180

## 4.8 采用ABACUS 进行表面计算（六)：补偿电荷

## 4.8.1介绍

现实中的化学反应多数时候发生在恒定电势条件下，而传统的DFT计算软件则是基于恒定电荷的条件。若要在计算模拟中保持恒定电势，在电化学表面反应建模时就需要调整晶胞中的电子数。在开源密度泛函理论软件 ABACUS 里，模拟体系时使用的是周期性边界条件，因此需要添加额外的算法保持模拟过程中同时保证超胞的电中性。其中的一种方案就是当从系统中添加/提取额外的电子时，需要在表面模型的真空区域中实现补偿电荷的分布。

目前 ABACUS 软件中实现了补偿电荷功能[1]，与增减电子数相对应的补偿电荷会以带电平板的形式（不考虑平板厚度）予以补偿进而保持电中性，并且存在相应的原子受力的修正，可以用于平面波（Plane-Wave）基组和数值原子轨道（Numerical Atomic Orbitals）基组计算，但注意目前该功能只能用于带真空层的表面体系。我们以下的例子以一个水分子为例，如图1所示。

<!-- image-->  
图1.一个水分子位于超胞中，红色代表氧原子，白色代表氢原子

## 4.8.2 文档与算例

ABACUS里关于补偿电荷的线上文档地址为：

http://abacus.deepmodeling.com/en/latest/advanced/scf/advanced.html#compensatin-charge

ABACUS 里关于补偿电荷参数的线上文档地址为:https:/abacus.deepmodeling.com/en/latest/advanced/input_files/inputmain.html#gate-field-compensating-charge

ABACUS 的补偿电荷算例下载地址为（gitee）:

https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/compensating_charge/H2O

可以采用的下载命令是：

1 git clone https://gitee.com/mcresearch/abacus-user-guide

之后进入 /abacus-develop/"examples/compensating_charge/H2O 目录

或者采用Github仓库地址：

https://github.com/MCresearch/abacus-user-guide/tree/master/examples/compensating_charge/H2O

## 4.8.3偶极修正与补偿电荷结合的表面计算

在采用ABACUS 进行表面计算（二)：偶极修正文档中，我们给出了水分子层计算中应用偶极修正之后的结果。在此基础上，我们可以进一步应用补偿电荷功能。如图1所示，我们采用了完全相同的一个水分子在周期性晶胞的结构，只有INPUT文件中添加了补偿电荷相关参数。

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix H20   
4 calculation scf   
5 nbands 8   
6 symmetry 0   
pseudo_dir ../../PP_ORB   
8   
9 #Parameters (2.Iteration)   
10 ecutwfc 100

11 scf_thr 1e-6   
12 scf_nmax 100   
13   
14 #Parameters (3.Basis)   
15 basis_type pw   
16   
17 #Parameters (4.Smearing)   
18 smearing_method gauss   
19 smearing_sigma 0.001   
20   
21 #Parameters (5.Mixing)   
22 mixing_type broyden   
23 mixing_beta 0.1   
24   
25 #Parameters (6.Dipole correction)   
26 efield_flag 1   
27 dip_cor_flag 1   
28 efield_pos_max 0.0   
29 efield_pos_dec 0.1   
30 efield_amp 0.0   
31   
32 #Parameters (7.Compensating charge)   
33 gate_flag 1   
34 zgate 0.7   
35 nelec 9   
36   
37 #Parameters (8.Output)   
38 out_pot 2  
以算例提供的INPUT文件为例，使用补偿电荷功能只需添加如下参数即可：

gate_flag：类型是Bool，指定补偿电荷功能的开(1)或者关(O)，默认值为0

·efield_dir：类型是Int，指定与带电平板垂直的方向，可取0，1，2分别表示x，y，z方向，默认值为2

·zgate：类型是Real，指定带电平板（位于真空中）在晶胞中的位置（分数坐标)，范围[0,1]，默认值为0.5

·nelec：类型是Real，设定的体系总电子数，这里是9个电子（氧原子因为质势有6个电子，每个氢原子有1个电子，总共8个电子)，即原体系多加了1个电子，那么为了保持电中性，带电平板会带1个正电荷

·block：类型是Bool，添加势垒以防止电子溢出的功能（可选)，1代表开，0代表关，默认值为0

block_down：类型是Real，势垒的起点（分数坐标)，默认是0.45

block_up：类型是Real，势垒的终点（分数坐标)，默认是0.55

block_height：类型是Real，势垒高度，单位是里德堡（Rydberg)，默认是0.1

## 4.8.4预期结果

如左下图所示，一个水分子位于超胞（尺寸为3318）的中部（即位于9A）。按照采用ABACUS 进行表面计算（二)：偶极修正文档中偶极修正参数设置的说明，锯齿即位于0.00.1的区域。注意在采用偶极修正/外加电场功能的情况下，带电平板的位置应该在真空区域，并且不同于偶极修正锯齿所在的区域。

为了验证结果的正确性，我们可以画出静电势沿Z轴（c方向）的变化如图2所示。与只应用偶极修正功能的静电势相比（可参考表面计算二里面的图3)，在分数坐标0.7（Z轴12.6埃）处，静电势出现了一个向下的凹

陷，这来自于此处的补偿电荷带电平板的影响。

静电势计算和作图方法请参考静电势和功函数计算文档。

<!-- image-->  
图2.静电势（Electrostatic Potential）沿超胞Z轴变化图

## 4.8.5参考文献

[1] Phys. Rev. B 89,245406 (2014), https://doi.0rg/10.1103/PhysRevB.89.245406

## 第5章特色功能

## 5.1ABACUS随机波函数DFT方法使用教程

## 5.1.1介绍

本教程旨在介绍 ABACUS 中随机波函数密度泛函理论（Stochastic Density Functional Theory,以下简称 SDFT）计算功能。目前 ABACUS 使用 SDFT主要聚焦在高温高压物质的模拟，特别是温稠密物质（Warm Dense Matter,简称WDM)。在进行温稠密物质计算时(温度高达数十到上千eV,1eV=11604.5K)，传统的Kohn-Sham密度泛函理论（KSDFT）需要用到极大数量的占据态电子波函数导致计算困难，而 SDFT使用随机波函数轨道，可以有效地避开对角化哈密顿矩阵这个问题，应用于高温计算。关于ABACUS 中实现 SDFT算法的细节可以参考QianruiLiuand Mohan Chen\*,"Plane-wave-based stochastic-deterministic density functional theory for extended systems," Phys.Rev.B,106,125132(2022)。本教程中将会展示如何在ABACUS 计算中使用 SDFT功能，此外还会介绍混合随机波函数密度泛函理论方法使用（mixed stochastic-deterministic DFT，简称 MDFT)，即在 SDFT 计算中，混入一部分的低能 Kohn-Sham 轨道，从而加速结果收敛。

## 5.1.2软件和算例准备

ABACUS 的软件包（3.2.0版本）中提供了一个SDFT的算例，可以从Gitee上下载。可以在网页右侧点击克隆/下载->下载 ZIP 得到算例，或者在linux 终端执行如下命令得到算例：

1 git clone htps://gitee.com/mcresearch/abacus-user-guide.git

下载后解压，之后进人/abacus-user-guide/examples/stochastic 文件夹。算例中有三个文件夹，pw_Si2、pW_md_Al和 186_PW_SDOS_10D10S。

## 5.1.3采用SDFT进行电子自洽迭代计算

pw_Si2文件夹：这是一个电子温度为0.6Ry（约8.16eV）的2个原子的金刚石结构硅（Si）的电子自洽迭代（Self ConsistentField，简称 SCF）算例，包含布里渊区K点的KPT文件和包含原子位置的 STRU文件与传统的 KSDFT计算并无区别，主要的不同在于输入文件 INPUT，注意目前 SDFT仅支持 smearing_method 为 fd。INPUT文件如下:

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 calculation scf   
4 esolver_type sdft   
5 pseudo_dir ../../PP_ORB   
6 nbands 4   
7 nbands_sto 64   
8 nche_sto 100   
9 method_sto 1   
10 #Parameters (Accuracy)   
11 ecutwfc 50   
12 scf_nmax 20   
13 symmetry 1

14 #Parameters (Smearing)   
15 smearing_method fd   
16 smearing_sigma 0.6

以上参数在ABACUS 的线上文档中均有详细说明，这里再进行简单概述：

calculation 设置为 scf，代表进行电子自治迭代计算（self-consistent feld)。

·esolver_type 是选择系统总能量求解方法的，默认为ksdft（Kohn-Sham density functional theory），这里需要设置为sdft才会使用 SDFT或者MDFT进行计算。

·nbands 是使用的Kohn-Sham 轨道的数目（也叫 deterministic orbitals，是通过严格对角化矩阵计算出来的）。如果nbands 设置为O，且nbands_sto（随机轨道数目，这里设成64）不为O，则会进行SDFT计算；如果nbands>0，且nbands_sto>0，则会进行混合KS 电子轨道和随机轨道（stochastic orbitals）的MDFT计算。注意：一般nbands 设置为能量低于费米能级对应的能带数，计算效率会比较高。

·nbands_sto是使用随机波函数（stochastic orbitals）轨道数目，原则上取得越大则随机误差越小，但计算效率也会相应降低。

·判断随机波函数个数是否足够的一个经验法则是：测试能量误差。实际计算中，一般可以采用10个左右不同的随机数种子（可以参考 seed_sto参数设置随机数种子，下面有介绍）生成的相同数目的随机波函数轨道进行 SDFT计算，得到系统能量并求得相应误差，增加随机波函数轨道数目直到控制能量误差小于万分之一即可。

当nbands_sto设为O时，程序会自动转成KSDFT进行计算（ABACUS3.2.2版本以后）。

·nche_sto 是将电子体系的哈密顿量进行切比雪夫展开的阶数，这个数取得越大则用到的切比雪夫展开阶数越多，相应的计算精度也会越高但效率会降低。大致关系为与温度成反比，温度越高，阶数可以取得越小；ecut（正比关系）越大，阶数越大；推荐使用的nche_sto的大小是使得输出文件 running_scf.log中的ChebyshevPrecision 小于 1e-8。

·method_sto是进行 SDFT计算使用的方法：1代表消耗内存较少但稍慢的方法，2代表更快但需要更大内存的方法，默认是2。

此外还有如下参数可能会用到：

·seed_sto:生成随机轨道的随机种子。默认是0，代表随时间随机生成；若要控制使用相同的随机种子，可以设置一个大于1的整数。

·bndpar:将所有并行的进程分成 bndpar个组，计算所用随机轨道将平均分布在每个组中，可以提高并行效率，默认为1。值得注意的是，这个参数并不是越大越好，并且不如K点并行（kpar 参数）有效，实际计算中应该优先使用K点并行，然后测试不同大小的bndpar，确定最佳的bndpar。

平面波能量截断值ecut 的测试：由于随机误差的出现，SDFT的ecut的测试与传统的 KSDFT稍有区别，但是原理是类似的。在确定好 nbands_sto后，就可以测试ecut了。与测试nbands_sto类似，在不同的ecut，需要采用10个左右不同的随机种子生成的相同数目的随机波函数轨道进行SDFT计算，然后取平均能量。由于温稠密物质能量一般都比较高，因此只要控制相邻二个ecut（相差为10Ry）对应的平均能量差小于一定标准即可（例如万分之一)。

注1：在这个例子里我们提供的赝势是Si.pz-vbc.UPF文件，这个文件包含4个硅的价电子。事实上，当温度特别高的时候，一般的赝势可能会面临可移植性差的问题，例如高温会使得内壳层电离。这个时候，要选择合理的赝势进行计算，甚至可能需要自己造一个新的赝势，目前ABACUS 3.2.0支持的是模守恒的赝势。

注2：ABACUS的SDFT和MDFT支持多个k点采样，因此可以在KPT文件里设置不同的k点个数，在某些性质的计算里，要注意计算性质随着k点的收敛。

## 5.1.4采用SDFT进行分子动力学模拟

pw_md_Al文件夹：这是一个电子温度为7.35Ry（约100eV）、包含16个铝（Al）原子的结构，我们对其进行分子动力学（MolecularDynamics，简称MD）的模拟。INPUT文件如下：

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 calculation md   
4 esolver_type sdft   
5 pseudo_dir ../../PP_ORB   
6 nbands 0   
7 nbands_sto 64   
8 nche_sto 20   
9 method_sto 2   
10 #Parameters (Accuracy)   
11 ecutwfc 50   
12 scf_nmax 20   
13 scf_thr 1e-6   
14 symmetry 1   
15 #Parameters (Smearing)   
16 smearing_method fd   
17 smearing_sigma 7.34986072   
18 #Parameters (MD)   
19 md_tfirst 1160400   
20 md_dt 0.2   
21 md_nstep 10

注意要进行分子动力学模拟，calculation 参数需设置为 md。esolver_type 需设置为 sdft，才能进行 SDFT的计算。这里nbands 设置为0，nbands_sto 设置为64，代表仅仅使用随机轨道而没有KS轨道的 SDFT计算。

## 5.1.5采用SDFT计算态密度

186_PW_SDOS_10D10S 文件夹：采用 SDFT还可以计算给定体系的态密度（Density of States，简称DOS）。例如，186_PW_SDOS_10D10S 是一个1个 Si原子的算例，电子温度约为0.6Ry（约8.16eV)。如下所示：

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix autotest   
4 calculation scf   
5 esolver_type sdft   
6 method_sto 2   
7   
8 nbands 10   
9 nbands_sto 10   
10 nche_sto 120   
11 emax_sto 0   
12 emin_sto 0   
13 seed_sto 20000   
14 pseudo_dir ../../PP_ORB

15 symmetry 1   
16 kpar 1   
17 bndpar 2   
18   
19 #Parameters (2.Iteration)   
20 ecutwfc 20   
21 scf_thr 1e-6   
22 scf_nmax 20   
23   
24 #Parameters (3.Basis)   
25 basis_type pw   
26   
27 #Parameters (4.Smearing)   
28 smearing_method fd   
29 smearing_sigma 0.6   
30   
31 #Parameters (5.Mixing)   
32 mixing_type broyden   
33 mixing_beta 0.4   
34   
35 out_dos 1   
36 dos_emin_ev -20   
37 dos_emax_ev 100   
38 dos_edelta_ev 0.1   
39 dos_sigma 4   
40 dos_nche 240   
41 npart_sto 2  
以上参数在ABACUS 的线上文档中均有说明，这里再进行简单概述：

。 out_dos:需要设置为1，才能输出能态密度。

dos_emin_ev:能态密度的能量最小范围，单位eV。

dos_emax_ev:能态密度的能量最大范围，单位eV。

dos_edelta_ev:控制输出DOS 时的能量间隔，单位eV。

dos_sigma:能态密度的高斯展宽的因子，单位eV。

dos_nche:计算能态密度时切比雪夫展开阶数，默认为100。

·npart_sto:当使用 method_sto=2 运行例如 DOS 的 SDFT 后处理时，将控制使用内存大小为正常的1/npart_sto，防止内存不够导致无法计算，默认为1。

注：态密度的输出文件是 OUT 文件夹下的 DOS1_smearing.dat。

## 5.1.6结尾

总体来讲，随机波函数密度泛函理论方法（SDFT或者MDFT）的使用与KSDFT并无太大的区别，直接运行ABACUS程序即可,但是对一些关键参数的选取会影响精度和效率(例如 nbands,nbands_sto,nche_sto,method_sto,kpar, bnd_par）。对于极端高温计算（>10eV），使用 SDFT可以大大提高计算速度，是比普通的 KSDFT 更好的选择。如果大家使用有问题，欢迎写信联系（见上)。

## 5.2ABACUS随机波函数DFT计算电子电导热导教程

## 5.2.1介绍

对于物质的电导热导，一般可以分成电子和离子贡献，而对于电子贡献的部分可以使用 Kubo-Greenwood 的公式进行计算[Phys.Rev.B83,235120 (2011)]，该方法已被广泛应用于液态金属、温稠密(Warm Dense matter,WDM)等物质的计算。Kubo-Greenwood通过直接计算含频的昂萨格系数 $L _ { m n } ( \omega )$ ，来计算电子的电导与热导：

$$
L _ { m n } ( \omega ) = ( - 1 ) ^ { m + n } \frac { 2 \pi e ^ { 2 } } { 3 \omega \Omega }
$$

$$
\times \sum _ { i j a k } W ( k ) \left( \frac { \epsilon _ { i k } + \epsilon _ { j k } } { 2 } - \mu \right) ^ { m + n - 2 } | \langle \Psi _ { i k } | \hat { v } _ { \alpha } | \Psi _ { j k } \rangle | ^ { 2 } [ f ( \epsilon _ { i k } ) - f ( \epsilon _ { j k } ) ] \delta ( \epsilon _ { j k } - \epsilon _ { i k } - \hbar \omega ) ,
$$

这里的n,m指标取值范围为1,2，Ω为体积, $W ( k )$ 为k点的权重， $\epsilon _ { i k }$ 为第i个波函数 $\Psi _ { i k }$ 对应的本征能量，$\begin{array} { r } { f ( \epsilon ) = \frac { 1 } { 1 + \exp \left( \frac { \epsilon - \mu } { k T } \right) } } \end{array}$ 为费米狄拉克分布， $\mu$ 为化学势， $\delta ( x )$ 为delta函数，在实际计算中delta函数可以用高斯函数

$$
G ( x ) = \frac { 1 } { \sqrt { 2 \pi s } } \exp \left( - \frac { x ^ { 2 } } { 2 s ^ { 2 } } \right)
$$

或洛伦兹函数 $\begin{array} { r } { L ( x ) = \frac { 1 } { \pi } \frac { \gamma } { x ^ { 2 } + \gamma ^ { 2 } } } \end{array}$ 代替， $\hat { v } _ { \alpha }$ 为速度算符的第一个分量 $( \alpha = x , y , z )$ 对于速度算符，速度算符的定义为 $\begin{array} { r } { \hat { v } = \frac { i } { \hbar } [ \hat { H } , \hat { r } ] } \end{array}$ 。

含频电导 $\sigma ( \omega ) = L _ { 1 1 } ( \omega )$ ，直流电导 $\begin{array} { r } { \sigma _ { 0 } = \operatorname* { l i m } _ { \omega \to 0 } \sigma ( \omega ) } \end{array}$

含频热导 $\begin{array} { r } { \kappa ( \omega ) = \frac { 1 } { e ^ { 2 } T } \left( L _ { 2 2 } - \frac { L _ { 1 2 } ^ { 2 } } { L _ { 1 1 } } \right) } \end{array}$ ，热导 $\kappa _ { 0 } = \operatorname* { l i m } _ { \omega \to 0 } ( \omega )$

以上便是传统 KSDFT计算的方法，通过 KSDFT求解的波函数、本征能量，可以轻易的带入公式进行计算。

对于随机波函数密度泛函理论(sDFT)，其没有波函数、本征能量不能通过该方法进行计算。而应从原始的Kubo 公式中出发：

电流、热流响应函数：

$$
C _ { m n } ( t ) = - 2 \theta ( t ) J \left\{ T r \left[ \sqrt { \hat { f } j _ { m } ( 1 - \hat { f } ) } e ^ { i \frac { \hat { H } } { h } t } j _ { n } e ^ { - i \frac { \hat { H } } { h } t } \sqrt { \hat { f } } \right] \right\} ,
$$

这里 $\theta ( x )$ 为阶跃函数，J为取虚部, $\begin{array} { r } { \hat { f } ~ = ~ \frac { 1 } { 1 + \exp \left( \frac { \hat { H } - \mu } { k T } \right) } } \end{array}$ 为费米狄拉克算符, $j _ { 1 } ~ = ~ e \hat { v }$ 为电流算符, $j _ { 2 } ~ =$ $( \hat { v } \hat { H } + \hat { H } \hat { v } ) / 2 - \mu \hat { v }$ 为热流算符, $e ^ { - i { \hat { H } } t / \hbar }$ 为演化算符。这里求迹便可通过随机波函数进行计算。

而昂萨格系数就可通过响应函数的傅里叶变换进行计算：

$$
L _ { m n } = \frac { 2 e ^ { m + n - 2 } } { 3 \Omega \hbar \omega } J [ \tilde { C } _ { m n } ( \omega ) ] , \vec { \gtrsim } \underline { { { \sf H } } } \underline { { { \sf t } } } + \vec { \ntrianglerighteq } \tilde { C } _ { m n } ( \omega ) \vec { \pi } ] \oslash \chi \{ \underline  { { \sf t } \} } \} \sharp \underline { { { \sf H } } } \mathcal { K } \lvert \overline { { { \sf H } } } \rvert \lVert \vec { \mu } \rVert \hat { \mu } ^ { \dagger } \hat { \lVert \tilde { \mu } \rVert } \vec { \lVert \tilde { \mu } \rVert } \hat { \ * } \hat { \lVert \tilde { \ * } \rVert } \hat { \ * } \hat { \lVert \tilde { \mu } \rVert } .
$$

高斯型:

$$
\tilde { C } _ { m n } ( \omega ) = \int _ { 0 } ^ { \infty } C _ { m n } ( t ) e ^ { - i \omega t } e ^ { - \frac { 1 } { 2 } s ^ { 2 } t ^ { 2 } } d t
$$

洛伦兹型：

$$
\tilde { C } _ { m n } = \int _ { 0 } ^ { \infty } C _ { m n } ( t ) e ^ { - i \omega t } e ^ { - r t } d t
$$

其分别与前面提到的高斯函数G(x)和洛伦兹函数L(x)对应。

## 5.2.2 算例准备

下载链接：

https://github.com/MCresearch/abacus-user-guide/tree/master/examples/stochastic/cond_Si

## 5.2.3采用sDFT进行电子的电导热导计算

## 输人文件:

cond_Si文件夹：这是一个电子温度为0.6Ry（约8.16eV）的单原子硅（Si）的电导热导计算算例，包含布里渊区K点的KPT文件和包含原子位置的 STRU文件与传统的KSDFT计算并无区别，主要的不同在于输入文件INPUT。为了计算电导热导，我们只需要把cal_cond参数打开即可，INPUT文件如下：

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix Si   
4 calculation scf   
5 esolver_type sdft   
6 nbands 10   
7 nbands_sto 40   
8 nche_sto 120   
9 seed_sto 20000   
10 symmetry 1   
11 kpar 1   
12 bndpar 2   
13   
14 #Parameters (2.Iteration)   
15 ecutwfc 40   
16 scf_thr 1e-6   
17 scf_nmax 100   
18   
19 #Parameters (3.Basis)   
20 basis_type pw   
21   
22 #Parameters (4.Smearing)   
23 smearing_method fd   
24 smearing_sigma 0.6   
25   
26 #Parameters (5.Mixing)   
27 mixing_type broyden   
28 mixing_beta 0.4   
29   
30 #Parameters (6.Conductivity)   
31 cal_cond 1   
32 cond_smear 1   
33 cond_fwhm 0.4   
34 cond_wcut 20   
35 cond_dw 0.02   
36 cond_dt 0.1  
以上参数在ABACUS 的线上文档中均有详细说明，这里再进行简单概述：  
。 cal_cond:控制是否计算电子贡献的电导热导  
。 cond_smear:设置展宽类型，1：高斯展宽（默认），2：洛伦兹展宽  
·cond_fwhm:设置半高宽FWHM，对于高斯展宽，FWHM=2√2ln2s；对于洛伦兹展宽FWHM=2γ

cond_wcut:计算频率的截断值，决定输出含频数据的频率范围，单位： $\mathtt { e V }$

。 cond_dw:频率的间隔，单位：eV

。cond_dt：对响应函数积分时的积分间隔，原子单位，响应函数振荡过快需要较小积分间隔，一般取值在0.01-0.2a.u.，在屏幕输出中有推荐的取值：

Recommended dt: 1.553e-01 a.u.

。 cond_dtbatch:其与 cond_dt 共同决定演化算符

$$
\exp ( i H * { \mathrm { c o n d } } _ { - } { \mathrm { d t b a t c h } } * d t )
$$

的切比雪夫展开阶数，一般来说该值设的越大，内存消耗越多，但速度越快，当其达到一定值，速度就不会增加。当其设为O时，程序会自动决定cond_dtbatch以使切比雪夫展开阶数在100 附近。计算时间的说明：

·使用 sDFT计算电导热导还是比较慢的，计算时间正比于切比雪夫展开阶数和

$$
n t = \frac { t _ { \mathrm { t o t a l } } } { ( \mathrm { c o n d \_ d t \_ b a t c h } * d t ) }
$$

。 切比雪夫阶数由平面波截断能决定，阶数约正比于截断能

nt由 $t _ { \mathrm { t o t a l } }$ 决定，而 $t _ { \mathrm { t o t a l } }$ 由 cond_fwhm决定，其长度反比于cond_fwhm，因此该方法不能使用任意小的展宽。

·增加cond_dt_batch\*dt 会使切比雪夫展开阶数增加，会使nt减小，总体效应是其值越大，计算效率越高，但达到一定程度，效率就不再增高了。

## 输出文件:

注：结果仅供参考，以最新版本算出的结果为准

屏幕输出:

1 set cond_dtbatch to 69 自动设置 cond_dtbatch 参数  
2 set N order of Chebyshev for KG as 113 演化算符的切比雪夫展开阶数  
3 Calculating conductivity....  
4 nw: 1000 ;dw: 2.000e-02 eV 总频率数，频率间隔  
5 nt：4862 ；dt：1.000e-01 a.u.(ry^-1）总时间数，时间间隔  
6 Emin_KS(2): 1.373e+01 eV; Emax: 5.442e+02 eV; Recommended max dt: 1.553e-01 a.u. 推荐 cond_dt  
7 ik=0：（Time left 1.673e+02 s）估计剩余计算时间  
8nt：计算进度  
9 69 138 207 276 345 414 483 552 621 690  
10 759 828 897 966 1035 1104 1173 1242 1311 1380  
11 1449 1518 1587 1656 1725 1794 1863 1932 2001 2070  
12 2139 2208 2277 2346 2415 2484 2553 2622 2691 2760  
132829 2898 2967 3036 3105 3174 3243 3312 3381 3450  
14 3519 3588 3657 3726 3795 3864 3933 4002 4071 4140  
15 4209 4278 4347 4416 4485 4554 4623 4692 4761 4830  
16  
17 DC electrical conductivity: 2.454778e+05 Sm^-1 直流电导值  
18 (直接线性外推的简单估计值，实际需要根据含频电导拟合外推)  
19 Thermal conductivity: 5.968738e+02 W(mK)^-1 热导值  
20 Lorenz number: 3.456403e+00 k_B^2/e^2 洛伦兹常数

je-je.txt:

<table><tr><td colspan="2">1 #t(a.u.)</td><td>c11(t)</td><td>c12(t)</td><td>c22(t)</td><td>decay</td></tr><tr><td colspan="2">2 0</td><td>10</td><td>5</td><td>10</td><td>1</td></tr><tr><td>3</td><td>0.1</td><td>-1.39981</td><td>6.61846</td><td>-38.6529</td><td>0.999999</td></tr><tr><td>4</td><td>0.2</td><td>-1.91869</td><td>6.94748</td><td>-25.4566</td><td>0.999997</td></tr><tr><td>5</td><td>0.3</td><td>-1.87453</td><td>5.46871</td><td>-15.4962</td><td>0.999993</td></tr><tr><td>6</td><td>0.4</td><td>-1.613</td><td>3.75367</td><td>-6.29163</td><td>0.999988</td></tr><tr><td>7</td><td>0.5</td><td>-1.36998</td><td>3.0228</td><td>-6.02995</td><td>0.999981</td></tr><tr><td>o</td><td>0.6</td><td>-1.1229</td><td>2.41076</td><td>-5.2121</td><td>0.999972</td></tr><tr><td>9</td><td>0.7</td><td>-0.84644</td><td>1.75119</td><td>-5.82033</td><td>0.999962</td></tr><tr><td>10</td><td>0.8</td><td>-0.529567</td><td>0.779473</td><td>-2.13103</td><td>0.99995</td></tr></table>

储存响应函数的文件，各列分别为均为原子单位，时间t,响应函数 $C _ { 1 1 } ( t ) , C _ { 1 2 } ( t ) , C _ { 2 2 } ( t )$ ,窗函数Onsager.txt:
<table><tr><td>1 ## w(eV)</td><td>sigma(Sm^-1)</td><td>kappa(W(mK)^-1)</td><td>L12/e(Am^-1)</td><td>L22/e^2(Wm^-1)</td></tr><tr><td>2 0.01</td><td>250820</td><td>604.318</td><td>-4.66249e+06</td><td>1.4392e+08</td></tr><tr><td>3 0.03</td><td>261503</td><td>619.205</td><td>-4.83345e+06</td><td>1.47997e+08</td></tr><tr><td>4 0.05</td><td>283291</td><td>649.006</td><td>-5.18082e+06</td><td>1.56228e+08</td></tr><tr><td>5 0.07</td><td>317019</td><td>693.79</td><td>-5.71545e+06</td><td>1.68767e+08</td></tr><tr><td>6 0.09</td><td>363915</td><td>753.684</td><td>-6.45332e+06</td><td>1.85835e+08</td></tr><tr><td>7 0.11</td><td>425574</td><td>828.87</td><td>-7.41519e+06</td><td>2.07723e+08</td></tr><tr><td>o 0.13</td><td>503909</td><td>919.555</td><td>-8.62601e+06</td><td>2.34774e+08</td></tr><tr><td>9 0.15</td><td>601083</td><td>1025.94</td><td>-1.01141e+07</td><td>2.67375e+08</td></tr><tr><td>10 0.17</td><td>719414</td><td>1148.17</td><td>-1.19101e+07</td><td>3.05944e+08</td></tr></table>

储存昂萨格系数的文件，各列分别为频率，电导，热导， $L _ { 1 2 } ( ) , L _ { 2 2 } ( )$

## 5.3ABACUS实时演化含时密度泛函理论使用教程

## 5.3.1 背景

实时演化含时密度泛函理论（Real-time Time-Dependent Density Functional Theory，简称 rt-TDDFT）是一种基于第一性原理的模拟激发态电子体系动态演化的方法。与传统的密度泛函理论（DFT）不同，rt-TDDFT 是建立在 Runge-Gross定理之上，主要研究时间依赖的 Kohn-Sham 方程（TDKS 方程)。该理论通过构建时间传播子迭代求解 TDKS 方程，直接演化电子的波函数。此外，它还采用半经典的Ehrenfest动力学来模拟离子实的运动，可以实时地得到激发电子态以及离子位置的信息，因此具有较为广泛的应用前景。

周期性体系具有能带结构，而非周期性的原子或分子体系则具有离散能级。体系的基态是指价电子从低到高依次填充至能带或能级中。当体系吸收特定能量后，低能态的电子跃迁至高能态，此时体系便处于激发态。研究激发态性质是TDDFT的一大优势，因此掌握如何模拟体系的激发态变得尤为关键。ABACUS 为此提供了两种方式：一种是直接改变占据数，使得叠加态的波函数拥有更多高能态分量；另一种是向一个处于基态的体系施加一个外加电场，为其提供激发所需的能量。下面将分别介绍这两种方式的具体使用。文档中提到的算例可以在这个仓库中找到。有关TDDFT输入参数的细节可参考输入参数线上文档

## 5.3.2在ABACUS中使用rt-TDDFT时改变电子的占据数

算例：htps://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/tddft/occupation算例输入文件 INPUT设置如下：

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix H2_ocp   
4 calculation md   
5 esolver_type tddft   
6 nbands 5   
7 nspin 1   
8 pseudo_dir ../../../tests/PP_ORB   
9 orbital_dir ../../../tests/PP_ORB   
10   
11 #Parameters (2.Iteration)   
12 ecutwfc 60   
13 scf_thr 1e-6   
14 scf_nmax 100   
15   
16 #Parameters (3.Basis)   
17 basis_type lcao   
18 gamma_only 0   
19   
20 #Parameters (4.Smearing)   
21 smearing_method gauss   
22   
23 #Parameters (5.MD Parameters)   
24 md_type nve   
25 md_nstep 1000   
26 md_dt 0.05   
27 md_tfirst 0   
28   
29 #Parameters (6.TDDFT Occupation Parameters)   
30 ocp 1   
31 ocp_set 11000

具体参数设置请参考线上文档，这里仅对部分重要参数进行说明：

·gamma_only：必须设置为0，gamma_only为1会采用double类型计算，rt-TDDFT使用传播子迭代计算波函数，需要complex 类型存储数据，不兼容 gamma_only为1的情况。

·basis_type：必须设置为1cao，目前 ABACUS 仅支持基于数值原子轨道基组的 rt-TDDFT功能。

·nbands：能带数，由于ocp_set 需要手动设置每一条能带的占据数到最高带，建议手动设置nbands，防止二者不对应。

·calculation：必须为md，同时一般MD(Molecular Dynamics)计算所需的参数都需要设置，这部分参数会对应rt-TDDFT的离子实运动的计算。

。 esolver_type：tddft，表示采用rt-TDDFT求解器。

。 ocp：改变占据数功能的开关，为1的时候开启。

·ocp_set：占据数的设置，1·1·0·0表示这个算例所计算的五条能带，各自占据的电子数，其总和为2，代表Hz的两个价电子，可以简化为2\*1·3\*0。另外，对于有多个k点的体系，ABACUS 支持对各个k点单独设置占据数。假设体系拥有5条能带，2个k点，则ocp_set 需要指定10个数值，前五个代表第一个k点五条能带的占据，后五个代表第二个k点五条能带的占据。

按照分子轨道理论，两个H原子的原子轨道发生相互作用形成一条成键轨道、一条反键轨道，H分子的基态占据可以表示为 $2 0 0 0 ^ { \ast } 0$ ，最低的能态代表成键轨道，则 $1 1 0 0 ^ { \ast } 0$ 占据数表示将一个电子从成键轨道上激发到反键轨道，会导致H分子发生解离。通过输出目录OUT.H2_ocp/STRU下的结构文件，可以看到随着时间的推移，两个H原子逐渐解离。运行occupation 文件夹下的算例，通过ASE-ABACUS 读取各个时刻 STRU 文件，并计算了两个H原子实时的间距d，结果如下图所示：

<!-- image-->  
为了说明这个解离是由于改变占据数引起的，可以将 INPUT 替换为occupation 文件夹中的 INPUT-1，其中ocp_set被改为: $2 0 0 0 ^ { \ast } 0$ ，同样输出实时的原子间距，可以看到，两个原子间距稳定的振荡，并未发生解离：

<!-- image-->

## 5.3.3ABACUS中使用rt-TDDFT外加电场计算材料吸收谱

在任意外加电场的影响下,rt-TDDFT能够实时模拟并获得材料的电子态和离子位置信息。按照 Runge-Gross定理（相当于KSDFT中的 Hohenberg-Kohn 定理)，这等价于材料的全部物理性质。在众多的物理性质中，吸收谱作为材料基本的光电响应性质，处理简单，结果直观，可参考对比的数据丰富，很适合用于介绍TDDFT 外加电场功能的基本细节。

## 规范选择

如何加入电场存在规范问题，根据电动力学的知识：

$$
E = - \nabla V - { \frac { \partial A } { \partial t } }
$$

第一种是长度规范，把电场的影响归为纯标势的影响，向势场中加入：

$$
V \propto E \cdot r
$$

因为晶胞很小，按照通常所用的光的波长，体系内的电场在空间上是近似匀强的，故这里没有写成积分的形式。这种形式相对简单，需要处理的项较少，程序处理较快，但是由于n的存在，它在周期性体系的边界是不连续的，只能采用锯齿状的方式在边界下降来衔接这种不连续性。对于非周期性体系，边界存在真空层，这种锯齿状衔接不会影响到材料的响应，但是对周期性体系，这种做法就行不通了。

对于周期性体系必须采用速度规范，它用矢势而非标势来引入电场：

$$
A ( t ) \propto \int _ { 0 } ^ { t } E ( t ^ { \prime } ) d t ^ { \prime }
$$

在这种规范下，需要对正常的项进行一系列变换，这里不详细论述，想进一步了解的可以参考文献2的内容。因此，我们实际上在程序里用的并不是直接的电场，而是标势或者矢势，但是它们都对应一个电场，程序指定电场的参数而不是势场的参数，这样可以统一处理长度和速度规范的场。

简单来说：长度规范只能处理非周期性体系，但是速度较快；速度规范可以处理周期性体系和非周期性体系，但是速度较慢，故一般建议对非周期性体系采用长度规范，周期性体系采用速度规范。

## 长度规范示例

算例：https://itee.com/mcresearch/abacus-user-guide/tree/master/examples/tddft/absorption_H2_length 输入文件INPUT:

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix H2_absoprion   
4 calculation md   
5 esolver_type tddft   
6 pseudo_dir ../../../tests/PP_ORB   
7 orbital_dir ../../../tests/PP_ORB   
8   
9 #Parameters (2.Iteration)   
10 ecutwfc 60   
11 scf_thr 1e-6   
12 scf_nmax 100   
13

14 #Parameters (3.Basis)   
15 basis_type lcao   
16 gamma_only 。   
17   
18 #Parameters (4.Smearing)   
19 smearing_method gauss   
20   
21 #Parameters (5.MD Parameters)   
22 md_type nve   
23 md_nstep 1000   
24 md_dt 0.005   
25 md_tfirst 0   
26   
27 #Parameters (6.Efield Parameters)   
28 td_vext 1   
29 td_stype 0   
30   
31 td_tstart 1   
32 td_tend 1000   
33   
34 td_vext_dire 33   
35 td_ttype 0。   
36 td_gauss_freq 3.66 1.22   
37 td_gauss_phase 0.00.0   
38 td_gauss_sigma 0.2 0.2   
39 td_gauss_t0 300 300   
40 td_gauss_amp 0.01 0.01   
41   
42 #Parameters (7.Output)   
43 out_chg 1   
44 out_efield 1   
45 out_dipole 1

其他的注意事项和占据数一样，这里注意补充一些电场参数的细节：

·md_nstep：MD步，这里为了做演示只用了1000步，演化总时长5fs，为了精度考虑，实际计算时应当增加步数，具体视所需体系有所差异，一般保证20fs以上总时长较为妥当。

。 td_vext：外加电场的开关，1为打开外加电场。

td_stype：规范的选择，0为长度规范，1为速度规范。

otd_tstart：开始加入电场的步数，在td_tstart 步之前，为无电场状态，一般为1即可。

·td_tend:电场结束的步数,td_tend后不再计算电场，用于无电场时节约资源，最大可以等同md_nstep的值。td_tend后，标势为O，而矢势固定为td_tend 处的值不再变化。

·td_type：电场的波包形状，0－3分别对应高斯、梯形、三角、阶跃等不同类型的电场。

·td_gauss_xxx：由于这里指定的是高斯型，对应高斯型的参数，具体可以参考输入参数的说明文档: https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html#tddft-time-dependent-density-functional-theory

。 out_dipole：输出每个时刻的电偶极矩，我们通过对电偶极矩的分析得到吸收谱。

。 out_efield：输出每个时刻的电场。

注意，这里为了覆盖更大的频率范围，通过对电场形式的参数多加一列的方式，在一次计算中加人了两个不同频率的电场。理论上可以用这种方式得到各种形状的场，当加的较多时，支持采用正则表达式 $m ^ { * } x$ 的形式进行简化。

运行算例，会输出全时刻的电偶极矩文件SPIN1_DIPOLE，如果nspin为2 则会有 $\mathtt { S P I N 2 \_ D I P 0 L E }$ 文件分别对应两种自旋的电子各自的电偶极矩；以及电场文件efield_x.dat，x的数量视所加的电场数量而定，这个算例就是两个：efield_O.dat和efield_1.dat。不考虑单位，通过下列公式得到分子的吸收谱信息：

$$
\alpha _ { i } ( \omega ) = \frac { \int { D _ { i } ( t ) e ^ { i \omega t } d t } } { \int { E _ { i } ( t ) e ^ { i \omega t } d t } }
$$

$D _ { i }$ 为方向的电偶极矩， $E _ { i }$ 为i方向的电场。程序自带了相应的后处理功能，具体请参考tools'/'plot-tools 下的说明进行使用，其结果如下：

<!-- image-->

这表明在z方向加的电场激发了一个频率为8.5eV左右的吸收峰。

## 速度规范示例

算例：ttps:/gitee.com/mcresearch/abacus-ser-guide/tre/master/examples/tddft/absorption_H2_velocity

速度规范的流程与长度规范基本相同，主要不同的参数有两个：

。 td_stype：变为了1，代表进行速度规范计算。

·out_current：输出电流的开关。对无限大的周期性体系无法定义电偶极，采用电流来代替，公式可以参考本文的参考文献2。简单来说：

$$
I ( t ) \propto \frac { d D ( t ) } { d t }
$$

因此在傅里叶变换时：

$$
D ( \omega ) \propto { \frac { I ( \omega ) } { i \omega } }
$$

---

## 第 201–220 页

因此二者可以得到类似的结果，不过current的结果由于分母的原因，在极低频时会无法避免的发散。相应的问题在文献[3]中有讨论。

## 5.3.4参考文献

[1] Meng S,Kaxiras E.Real-Time，Local Basis-Set Implementation of Time-Dependent Density Functional Theory for Excited State Dynamics Simulations[J].The Journal of Chemical Physics，2008，129(5), https://doi.0rg/10.1063/1.2960628.

[2] Pemmaraju CD, Vila FD,Kas JJ,et al. Velocity-Gauge Real-Time TDDFT Within a Numerical Atomic Orbital Basis Set[J]. Computer Physics Communications,2018,226: 30-38,htps://doi.org/10.1016/j.cpc.2018.01.013.

[3] Yabana K， Sugiyama T, Shinohara Y,et al. Time-DependentDensity Functional Theory forStrong Electromagnetic Fieldsin Crystalline Solids[J].Physical Review B，2012，85(4):045134, https://link.aps.org/doi/10.1103/PhysRevB.85.045134.

## 5.4ABACUS无轨道密度泛函理论方法使用教程

## 5.4.1无轨道密度泛函理论背景知识

无轨道密度泛函理论(Orbital free density functional theory,OFDFT)是一种第一性原理模拟方法,相比于 KohnSham DFT(KSDFT)，它的优势之一在于O(NInN)的算法复杂度，这使得 OFDFT可以用于上万原子甚至更大体系的电子基态计算，或者大体系、长时间的第一性原理分子动力学等。

目前，OFDFT已被应用于简单金属、合金、半导体、小分子、温稠密物质等体系。

## 无轨道密度泛函理论

在OFDFT的框架下，体系的总能量泛函可以写为

$$
\begin{array} { r } { E _ { O F } [ \rho ] = T _ { s } [ \rho ] + E _ { e x t } [ \rho ] + E _ { \Pi } [ \rho ] + E _ { x c } [ \rho ] + E _ { \Pi } . } \end{array}
$$

依次为无相互作用动能 $T _ { s } [ \rho ]$ ，电子-离子相互作用能 $E _ { e x t } [ \rho ]$ ，电子-电子相互作用能 $E _ { \Pi } [ \rho ]$ ，交换关联能 $E _ { x c } [ \rho ]$ 离子-离子相互作用能 $E _ { \Pi }$ ，其中 $\rho$ 为电荷密度。

为了在粒子数守恒的条件下求解其极小值，利用拉格朗日乘子法，定义

$$
L = E _ { O F } [ \rho ] - \mu \left( \int \rho ( r ) d r - N \right) ,
$$

可以证明，这里的乘子 $\mu$ 就是无相互作用体系中最富占据态的能量，即化学势。为了保证密度处处为正，一般对 $\sqrt { \rho }$ 进行优化，因此求L对 $\sqrt { \rho }$ 的导数，为了方便，定义

$$
\phi ( r ) = \sqrt { \rho ( r ) } ,
$$

则有

$$
{ \frac { \delta L } { \delta \phi } } = { \frac { \delta E _ { O F } [ \rho ] } { \delta \phi } } - 2 \mu \phi = 0 \to V ( r ) \phi ( r ) = \mu \phi ( r ) .
$$

(\*）

其中 $\begin{array} { r } { V ( r ) = \frac { \delta E _ { O F } [ \rho ] } { \delta \rho ( r ) } } \end{array}$ 为势能。这就是OFDFT求解的方程，一般用共轭梯度（CG）法、截断牛顿（TN）法或L-BFGS 等优化算法求解。目前ABACUS 中实现了TN 法和两种CG 法（Polak-Ribire 形式和 Hager-Zhang 形式)，默认采用TN法。

ABACUS 基于平面波基矢量，实现了上述流程，可以进行基于OFDFT的自洽计算，分子动力学计算，以及结构驰豫。

## 动能泛函

OFDFT 的精度高度依赖于动能泛函（kinetic energy density functional,简称 KEDF）的精度，目前 ABACUS中实现了Thomas-Fermi (TF)[1],von Weizsacker (vW)[2],TFAvW [3], Wang-Teter (WT)[4],Luo-Karasiev-Trickey(LKT)[5]共五种动能泛函。

下面我们对这些泛函做简单介绍，并且介绍在 ABACUS 的INPUT文件中如何设置相关的参数。Thomas-Fermi KEDF:

设置INPUT文件中的of_kinetic tf参数

$$
T _ { T F } = \frac { 3 } { 1 0 } ( 3 \pi ^ { 2 } ) ^ { 2 / 3 } \int \rho ^ { 5 / 3 } ( r ) d r ,
$$

对均匀电子气精确成立，可用于极高温体系，比如处于温稠密状态的金属。可通过 $\mathsf { o f \_ t f _ { \mathrm { - } } w e i g h t }$ 调整其权重,默认为1。

von Weizsäcker KEDF:

设置of_kinetic vw

$$
T _ { v w } = - \frac { 1 } { 2 } \int \sqrt { \rho ( r ) \nabla ^ { 2 } } \sqrt { \rho ( r ) } d r ,
$$

对单电子、双电子体系（只有一条轨道）严格成立，一般不单独使用。可通过 $0 \mathbf { f } _ { - } \mathtt { V W \_ W e i g h t }$ 调整其权重，默认为1。

TFλvWKEDF:

设置of_kinetic tf+

$$
T = T _ { T F } + \lambda T _ { v w } ,
$$

当 $\begin{array} { r } { \lambda = \frac { 1 } { 9 } } \end{array}$ 时就是TFKEDF的二阶梯度展开，一般 $\textstyle \lambda = { \frac { 1 } { 5 } }$ 时表现最好。

参数入可通过of_vw_weight 设置，默认为1。

Wang-Teter KEDF:

设置of_kinetic wt

$$
T _ { W T } = \frac { 3 } { 1 0 } ( 3 { \pi ^ { 2 } } ) ^ { 2 / 3 } \iint { \rho ^ { \alpha } ( r ) W ( r - r ^ { \prime } ) \rho ^ { \beta } ( r ^ { \prime } ) d r d r ^ { \prime } } + T _ { v W } + T _ { T F } ,
$$

基于Lindhard响应函数推导，在简单金属Li、Mg、Al中有着不错的表现。是ABACUS 默认采用的 KEDF。参数 $\alpha , \beta$ 可通过 $\mathsf { o f \_ w t \mathsf { _ { - } a l p h a } }$ 和 $\mathsf { o f \_ w t \_ b e t a }$ 设置，默认值均为 $\frac { 5 } { 6 }$ 。

Luo-Karasiev-Trickey KEDF:

设置of_kinetic lkt

$$
T _ { L K T } = \int T _ { T F } \frac { 1 } { \cosh \alpha s } d \boldsymbol { r } + T _ { v W } , \quad s = \frac { 1 } { 2 ( 3 \pi ^ { 2 } ) ^ { 1 / 3 } } \left| \nabla \rho \right| \rho ^ { 4 / 3 } .
$$

可用于简单金属和半导体，计算效率较高，但在简单金属中精度低于WTKEDF。

参数α可通过of_lkt_a设置，默认值为1.3。

## 局域赝势

由于OFDFT中舍弃了单电子轨道，无法采用常用的非局域赝势，如模守恒赝势，而必须采用局域赝势。

目前 ABACUS 支持 BLPS (bulk-derived local pseudopotential)。

下载地址：htps://ithub.com/EACcodes/local-pseudopotentials 实空间鹰势：ABINIT,ABACUS；倒空间势:PROFESS。势生成(需要和 ABINIT 7.0.5 结合): https://github.com/EACcodes/BLPSGenerator 覆盖Li,Mg,Al, Si,P,Ga,As,In, Sb 九种元素

使用BLPS 时，需要在ABACUS 里调整的参数有：

INPUT 中：pseudo_rcut 16

STRU 中：赝势种类设置为blps，比如Al 26.98 al.lda.lps blps

## 5.4.2ABACUS中进行OFDFT计算的具体流程

## 自洽计算

示例

下面是输入文件的示例：

INPUT文件记录OFDFT计算所需主要参数

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix example   
4 calculation scf   
5 esolver_type ofdft   
6   
symmetry1   
8 pseudo_dir../../PP_ORB/   
9 pseudo_rcut 16   
10 nspin 1   
11   
12 #Parameters (2.Iteration)   
13 ecutwfc 60   
14 scf_nmax 50   
15   
16 #Parameters (3.Basis)   
17 basis_type pw   
18   
19 #OFDFT   
20 of_kinetic wt   
21 of_method tn

STRU 文件记录元素种类、质量、赝势，晶格矢量，原子坐标等信息

```csv
1 ATOMIC_SPECIES
2 Al 26.98 al.lda.lps blps
3
4 LATTICE_CONSTANT
5 7.50241114482312 // add lattice constant
6
7 LATTICE_VECTORS
80.000000000000 0.500000000000 0.500000000000
9 0.500000000000 0.000000000000 0.500000000000
10 0.500000000000 0.500000000000 0.000000000000
11
12 ATOMIC_POSITIONS
13 Direct
14
15 Al
16 0
171
```

18 0.0000000000000.00000000000.000000000000111

KPT 文件（因为OFDFT没有电子波函数，所以不需要布里渊区的多个k点，Gamma 点就可以）

```csv
1 K_POINTS
2 0
3 Gamma
4111000
```

如上所示，与KSDFT的自治计算相比，OFDFT 自治计算的输入文件有以下几个区别：

## 。 INPUT

·不需要设置 smearing 和charge mixing 相关参数，如果设置了也没有关系，这些参数不会影响 OFDFT计算;

将esolver_type设置为ofdft;

将 pseudo_rcut 设置为16，以适配 BLPS 优势。

## 。 STRU

将优势种类设置为blps。

做完以上调整后，即可使用默认参数进行OFDFT的自治计算。

下面列举一些其它的重要参数:

·of_kinetic：用于选择功能泛函，可选项有tf,vw,tf+,wt,lkt，默认值为wt，具体介绍见1.2节;

·of_method：用于选择优化方法，可选项有tn,cg1,cg2，分别对应截断牛顿法和两种CG法（Polak-Ribire形式和Hager-Zhang 形式)，默认为 tn。一般而言，效率上tn >cg2>cg1;

·of_full_pw：做快速傅里叶变换（FFT）时，是否使用全部的平面波，默认为True。建议打开，可以保证计算的稳定性和精度;

·of_full_pw_dim:控制FFT维数的奇偶性，可选项有O,1,2,分别表示可奇偶，保证为奇数，保证为偶数,默认为0。FFT维数为偶数时，可能导致微小的误差，但一般来说可以忽略。需要注意的是，如果打开了nbspline，则需要设置of_full_pw_dim1，否则会导致计算不稳定。

## 注意事项

目前 ABACUS 的OFDFT模块并不是十分完善，使用时请注意以下几个注意事项：

目前 OFDFT不支持 gamma only，因此使用 OFDFT功能时请关闭 gamma_only;

目前OFDFT只支持自旋简并，即nspin1的计算;

·如果使用 PBE 泛函，建议用dft_functional XC_GGA_X_PBE+XC_GGA_C_PBE调用Libxc 中的PBE，否则可能导致计算不稳定。

## 分子动力学与结构弛豫

ABACUS 中支持使用OFDFT作为能量、力和应力的求解器，进行分子动力学模拟与结构弛豫。

与使用KSDFT进行分子动力学或结构弛豫相比，使用OFDFT时，不需要对MD，relax，或cel-relax 相关参数进行修改，只需要按照2.1中的方式，将能量、力和应力的求解器替换为OFDFT。

下面是几个实际的INPUT例子：

分子动力学（MD)

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix test   
4 calculation md   
5 esolver_type ofdft

## 原子结构弛豫(relax)

## 晶格弛豫（cell-relax）

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix test   
4 calculation cell-relax   
5 esolver_type ofdft   
6 pseudo_dir ../../PP_ORB   
7 pseudo_rcut 16   
8   
9 #Parameters (2.Iteration)   
10 ecutwfc 60   
11 scf_nmax 100   
12   
13 #0FDFT   
14 of_kinetic wt   
15 of_method tn   
16   
17 #Parameters (3.Basis)   
18 basis_type pw   
19   
20 relax_nmax 50

## 5.4.3参考文献

[1]Fermi E.Statistical method to determine some properties of atoms[J].Rend.Accad.Naz.Lincei,1927,6(602- 607): 5.

[2] Weizsäcker C F. Zur theorie der kernmassen[J]. Zeitschrift fur Physik,1935, 96(7-8): 431-458.

[3] Berk A.Lower-bound energy functionals and their application to diatomic systems[J]. Physical Review A,1983, 28(4): 1908.

[4] Wang L W,Teter MP. Kinetic-energy functional of the electron density[J]. Physical Review B,1992,45(23): 13196.

[5]Luo K,Karasiev V V,Trickey S B.A simple generalized gradient approximation for the noninteracting kinetic energy density functional[J]. Physical Review B,2018, 98(4): 041111.

## 5.5ABACUS隐式溶剂模型使用教程

## 5.5.1介绍

近年来，电化学表面反应，例如一些电催化系统，无论是在工业界还是科学研究领域，都受到了广泛的关注。而对于一个真实电极表面反应体系，往往需要考虑许多复杂的情况，例如电极与电解液之间的固液界面，在材料模拟中经常遇到和使用。在这种系统的第一性原理计算中应该考虑溶剂化效应，以便获得准确的性质预测。

溶剂化是指溶质和溶剂之间的相互作用，使溶质在溶液中稳定。这种相互作用的强度和性质影响体系的许多特性。在计算化学中，溶剂模型(Solvationmodel)是描述溶剂化行为的一种计算方法。

隐式溶剂化模型(Implicit solvation model)是一种处理溶剂化效应的成熟方法，在有限系统和周期系统中得到了广泛的应用。这种方法将溶剂视为连续介质，而不是单独的“显式”溶剂分子，这意味着嵌入隐式溶剂中的

溶质以及溶剂自由度的平均值将隐含在溶剂浴的性质中。与“显式”方法相比，这种隐式溶剂化模型可以以低得多的计算成本提供定性正确的结果，特别适用于大型复杂系统。

ABACUS 中实现的隐式溶剂化模型采用了Mathew、Sundararaman、Letchworth-Waver、Arias 和 Hennig 在2014 年开发的方法[1]。

## 5.5.2 文档和算例

ABACUS 里关于隐式溶剂模型的线上文档地址为：http://abacus.deepmodeling.com/en/latest/advanced/scf/advanced.html#implicit-solvation-model

ABACUS里关于隐式溶剂模型参数的线上文档地址为：

https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html#   
implicit-solvation-model ABACUS 的隐式溶剂模型算例下载地址为（gitee):

https://gitee.com/deepmodeling/abacus-develop/tree/develop/examples/32_imp_sol_model/   
/Pt-slab

可以采用的下载命令是：

1 git clone htps://gitee.com/deepmodeling/abacus-develop.git

之后进人/abacus-develop/examples/32_imp_sol/Pt-slab 目录

或者采用Github仓库地址：

https://github.com/deepmodeling/abacus-develop/tree/develop/examples/32_imp_sol_model/ Pt-slab

## 5.5.3输人参数

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix h2   
4 calculation scf   
5 ntype 1   
6 nbands 2   
7 symmetry   
8 pseudo_dir ·/   
9   
10 #Parameters (2.Iteration)   
11 ecutwfc 60   
12 scf_thr 1e-6   
13 scf_nmax 100   
14   
15 #Parameters (3.Basis)   
16 basis_type pw   
17   
18 #Parameters (Solvation Model)   
19 imp_sol 1   
20 eb_k 80   
21 tau 0.000010798   
22 sigma_k 0.6

<table><tr><td>23 nc_k 0.00037</td></tr></table>

以算例提供的INPUT文件为例，使用隐式溶剂模型方法只需添加如下参数即可：

。 imp_sol：类型是Bool，指定隐式溶剂模型的开(1)或者关(O)，默认值为0

。 eb_k：类型是Real，溶剂的相对介电常数，水为80

·tau：类型是Real，有效表面张力参数，用于描述未被静电项捕获的溶质和溶剂之间的空化、分散和排斥相互作用，单位是 $R y / B o h r ^ { 2 }$ ,默认值为1.0798e-05

·sigma_k：类型是Real，由溶质的电子结构隐含地确定的扩散腔的宽度（溶质电子密度与nc_k的比值，无量纲)，默认值为0.6

·nc_k：类型是Real，介电腔形成时的电子密度值，单位是 $B o h r ^ { 3 }$ ，默认值为0.00037

注意，eb_k可根据需要设置为模拟溶剂分子的相对介电常数参数，默认值80为水分子情况。tau，sigma_k,$\mathtt { n c \_ k }$ 的默认值均由在水溶液情况下与实验结果拟合得到，如需考虑其他溶剂，可自行测试拟合。

<!-- image-->

## 5.5.4预期结果

如图所示，一个氢分子位于超胞中，模拟其处于溶液环境时的状态。将imp_sol设为1，其他参数使用默认值，表示溶剂为水。在输出文件OUT.ABACUS/running_scf.log中，E_sol_el和E_sol_cav分别表示静电作用和空腔作用引起的能量修正。如案例所示，E_sol_el代表溶质和溶剂分子之间的静电感应作用，贡献的能量修正一般为负值。E_sol_cav 代表溶质分子在溶剂中产生空腔所引起的能量修正，一般为正值。

体系溶剂化能的计算：打开隐式溶剂模型(imp_sol=1)时体系总能量(结构优化之后)与真空情况(imp_sol=0)体系总能量（结构优化之后）之差。

<!-- image-->

<table><tr><td>Energy</td><td>Rydberg</td><td>eV</td></tr><tr><td>E_KohnSham</td><td>-2.33128015246</td><td>-31.7186937078</td></tr><tr><td>E_Harris</td><td>-2.3318039001</td><td>-31.72581966</td></tr><tr><td>E_band</td><td>-1.48509692705</td><td>-20.2057802902</td></tr><tr><td>E_one_elec</td><td>-2.78774594423</td><td>-37.9292294179</td></tr><tr><td>E_Hartree</td><td>+1.52957079998</td><td>+20.8108783741</td></tr><tr><td>E_XC</td><td>-1.37573099267</td><td>-18.7177804156</td></tr><tr><td>E_Ewald</td><td>+0.302102236829</td><td>+4.11031179942</td></tr><tr><td>E_demet</td><td>+0</td><td>+0</td></tr><tr><td>E_descf</td><td>+0</td><td>+0</td></tr><tr><td>E_exx</td><td>+0</td><td>+0</td></tr><tr><td>E_sol_el</td><td>-0.00119898588563</td><td>-0.0163130398661</td></tr><tr><td>E_sol_cav</td><td>+0.00172273352515</td><td>+0.0234389920776</td></tr><tr><td>E_Fermi</td><td>-0.742548463525</td><td>-10.1028901451</td></tr></table>

## 5.5.5参考文献

[1] J. Chem. Phys.140, 084106 (2014), https://doi.0rg/10.1063/1.4865107

## 5.6ABACUS+DPGEN使用教程

## 5.6.1介绍

本教程旨在介绍结合ABACUS（中文名原子算筹，这里基于ABACUS 3.2.0版本）和DPGEN软件结合生成机器学习DP势，这里ABACUS 扮演了提供第一性原理训练数据的作用。本教程以碳化硅（SiC）材料为案例，

并且教大家使用 DPGEN autotest功能，计算 SiC 的 Equation of State (EOS)和弹性常数等性质。先介绍主要软件和计算平台：

DPGEN 软件：是一个结合第一性原理软件来自动产生训练数据标签的软件，如果不熟悉 DPGEN 的读者，这里可以推荐大家先阅读DPGEN的相关原理文档和说明，熟悉这部分的读者可以跳过：

https://docs.deepmodeling.com/projects/dpgen/en/latest/   
https://bohrium-doc.dp.tech/docs/software/DP-GEN   
https://bohrium-doc.dp.tech/docs/software/DP-GEN_init   
https://bohrium-doc.dp.tech/docs/software/DP-GEN_autotest   
https://www.sciencedirect.com/science/article/pii/S001046552030045X?via%3Dihub

计算平台：由于本案例通过 Bohrium 平台教学，所以所用到的 machine.json 文件仅限于 Bohrium 平台使用,如需在其他平台上使用，可参考下面网址，修改machine.json文件：

https://docs.deepmodeling.com/projects/dpgen/en/latest/run/example-of-machine.html

## 5.6.2准备

本教程用到的所有输入文件，均可点击这里下载。

(https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/abacus-dpgen)

此外，在Linux下可以采用的下载命令是：

1 git clone https://gitee.com/mcresearch/abacus-user-guide.git

## 案例下载地址

之后进入/abacus-user-guide/examples/md目录

或者采用Github仓库地址：

https://github.com/MCresearch/abacus-user-guide/tree/master/examples/md

下载完之后，应该可以看到三个文件夹：

init，run，以及auto-test

init文件夹：包含用于dpgen init_bulk步骤所需的输入文件，用户可参考其设置自己的体系参数。

run 文件夹：包含用于dpgen run 步骤所需的输入文件，用户可参考其设置自己的体系参数。

auto-test文件夹：包含用于计算弹性常数等性质的输入文件，用户可参考其设置自己的体系参数。

## 安装相关软件

这里默认读者已经安装好了ABACUS 软件,本算例中ABACUS 计算采用的是数值原子轨道的LCAO(LinearCombination of Atomic Orbitals）方法，因为一般来说LCAO会比平面波效率更高，此外读者也可以把密度泛函理论计算部分改成平面波基矢量 (后续会讲到)。

开始之前，请务必安装/更新dpgen 和dpdata（用来转换数据）到最新版本：

1.安装/更新 dpdata pip install -user git+"htps://gitee.com/deepmodeling/dpdata.git@devel

2.安装/更新 dpgen pip install -user git+"https://gitee.com/deepmodeling/dpgen.git@devel

## 5.6.3结合DPGEN生成DP数据的流程

DPGEN使用流程如下，接下来我们将一一展开介绍：

<!-- image-->

## 准备赝势和数值原子轨道文件

本例子所需文件有：

模守恒赝势文件，格式为\*.upf,也可以使用其他ABACUS 支持的赝势格式，注意ABACUS 目前不支持PAW方法或者超软赝势。

数值原子轨道文件，格式为\*.orb，注意这里的数值原子轨道文件要和赝势文件匹配，也就是数值原子轨道是由指定赝势产生的，关于如何生成数值原子轨道，我们会提供中文教程。

注：在ABACUS官网上可以下载到赝势和数值原子轨道文件：

http://abacus.ustc.edu.cn/pseudo/list.htm

本例采用的赝势和轨道是：

C 的 PBE 模守恒 ONCV 模守恒赝势：C_ONCV_PBE-1.0.upf

Si 的 PBE 模守恒 ONCV 模守恒赝势：Si_ONCV_PBE-1.0.upf

C 的PBE模守恒ONCV赝势对应的数值原子轨道：C_gga_9au_100Ry_3s3p2d.orb

Si 的 PBE 模守恒ONCV赝势对应的数值原子轨道：Si_gga_9au_100Ry_3s3p2d.orb

注：如果是使用平面波基矢量来做密度泛函理论计算，则不需要提供数值原子轨道文件，结合DPGEN 使用的流程是一致的（仅需要删除调用轨道文件的输入参数即可)。如果采用其他ABACUS 支持的赝势文件，可自行生成轨道文件（请参考本教程系列的其它相关中文教程）：

注：这里C_gga_9au_100Ry_3s3p2d.orb的意思是C元素、采用GGA泛函、轨道截断半径是9a.u.（轨道截断半径越大，则ABACUS 所需求解 Kohn-Sham 方程的时间会越长，而且这个时间对于轨道半径极其敏感，推荐可以测试不同轨道半径，选取合适的、截断半径小的轨道可以加速计算)，100Ry是推荐的平面波截断值（哪怕用了数值原子轨道，程序里还是会用平面波来做某些物理量的计算，比如模守恒赝势的局域赝势部分)，3s3p2d表示3个s径向轨道、3个p轨道、2个d轨道。

## 准备原子体系的结构文件

本步骤所需的描述原子结构的文件有：

\*.cif

可以从多个渠道下载所需体系的结构文件，本案例主要从materials project 网站下载cif文件，使用的 SiC 对应多个原子构型，主要有3C、2H、4H、6H四个构型，采用哪些构型取决于用户所研究的体系和性质，一般可选取所关心的物质温度和压强区间内可能出现的结构。本案例主要采用3C和2H构型来演示。

3C-SiC: https://materialsproject.org/materials/mp-8062?formula=SiC  
2H-SiC: https://materialsproject.org/materials/mp-7140?_skip=15&formula=SiC  
4H-SiC: https://materialsproject.org/materials/mp-11714?formula=SiC  
6H-SiC: https://materialsproject.org/materials/mp-7631?formula=SiC  
得到了cif之后，可以使用下列方法把cif 转换成 STRU文件：  
1.安装 ASE-ABACUS 接口：

1 git clone https://gitlab.com/1041176461/ase-abacus.git   
2 cd ase-abacus   
3 python3 setup.py install

2.设置环境变量(可选)

ABACUS 支持两种基组：平面波（PW）和数值原子轨道（LCAO)。赝势和轨道文件的存放路径可以通过环境变量设置，分别为：ABACUS_PP_PATH和ABACUS_ORBITAL_PATH,设置方法如下:

```shell
1 PP=\${HoME}/pseudopotentials
2 ORB=\${HOME}/orbitals
3 expOrt ABACUS_PP_PATH=\${PP}
4 expOrt ABACUS_ORBITAL_PATH=\${ORB}
```

PW 计算只需要设置 ABACUS_PP_PATH。LCAO 需要两个都设置：ABACUS_PP_PATH and ABA-CUS_ORBITAL_PATH。

3.CIF 转 STRU

```python
from ase.io import read, write
2 from pathlib import Path
3
4 cs_dir = './'
5 cs_vasp = Path(cs_dir，'SiC.cif'）\# 修改为自已的 cif 文件名
6 cs_atoms = read(cs_vasp, format='cif')
cs_stru = Path(cs_dir，'STRU')
8 Pp = {'Si':'Si_ONCV_PBE-1.0.upf','C':'C_ONCV_PBE-1.0.upf'} \# 修改为所对应的赝势文件名
9 basis = {'Si':'Si_gga_9au_100Ry_3s3p2d.orb','C':'C_gga_9au_100Ry_3s3p2d.orb'}\#修改为所对应的轨道文件名
10 write(cs_stru, cs_atoms, format='abacus', pp=pp, basis=basis)
```

运行上述 python 脚本，把cif转成 ABACUS 所需的 STRU 文件。

## 产生初始训练数据

本步骤所需文件有：

原子结构文件：STRU

数值原子轨道文件：\*.orb

赝势文件：\*.upf

布里渊区k点采样：KPT

提交任务的文件：machine.json

DPGEN 的输入参数文件：param.json

为每一个构型新建一个名为init的文件夹。进人该文件夹后，为每个构型都新建一个单独的文件夹，之后单独运行 dpgen init_bulk命令，用于产生接下来的dpgen run 步骤的初始数据集。本案例所提供的初始原子构型为3C 和 2H结构，并把赝势文件、轨道文件、STRU文件放入对应的文件夹里。按照下面链接的案例准备适合自己机器的 machine.json文件，并分别放入3C和2H文件夹。

https://docs.deepmodeling.com/projects/dpgen/en/latest/run/example-of-machine.html 本教程所用到的文件可以从Gitee下载，修改为符合自己的配置即可。

接下来准备 param.jsion 文件，本步骤最关键的设置也是 param.json 文件设置，案例如下：

1{   
2 "init_fp_style": "ABACUS",   
3 "stages": [1,2,3,4],   
4 "cell_type": "diamond",   
5 "super_cell": [1，1，1],   
6 "elements": ["Si","C"],

```jsonl
"from_poscar": true,
8 "from_poscar_path": "./STRU",
9 "potcars": ["Si_ONCV_PBE-1.0.upf","C_ONCV_PBE-1.0.upf"],
10 "orb_files": ["Si_gga_9au_100Ry_3s3p2d.orb","C_gga_9au_100Ry_3s3p2d.orb"],
11 "relax_incar": "./INPUT_rlx",
12 "md_incar" : "./INPUT_md",
13 "relax_kpt": "./KPT",
14 "md_kpt": ". /KPT",
15 "md_nstep": 10，
16 "scale": [0.99,1.00,1.01],
17 "skip_relax": false,
18 "pert_numb": 10，
19 "pert_box": 0.03,
20 "pert_atom": 0.01,
21 "deepgen_templ": "../generator/template/",
22 "coll_ndata": 5000，
23 "_comment " : "that's all"
24}
```

init_fp_style 关键字需设置为 ABACUS，如果from_poscar为true，则会读取 from_poscar_path 指定的 STRU文件，且cell_type 关键字失效。super_cell用于括胞，推荐直接使用括胞之后的 STRU 文件，则 super_cell 设置为[1,1,1]。括胞一般根据需求而定，使得三个方向的cell长度约为6埃以上（经验性的)。其中还用到了\*."upf和\*."orb文件，修改为自己所计算的体系即可。

init_bulk的基本操作流程分为以下4个顺序步骤：

生成初始(平衡)结构：输出文件存储在00.place_ele。

产生 MD 起始构象：放缩扰动00.place_ele 中的结构，输出文件存储在01.scale_pert。

执行 AIMD：取01.scale_pert起始构象调用 vasp 执行少步数 AIMD，输出文件存储在 02.md。

整理初始数据：调用dpdata提取02.md中 AIMD各 snapshots 的数据作为初始数据，输出文件存储在 02.md。

param.json中将使用 stages 关键词的取值14控制以上4步的执行，操作中，各步骤(2,3,4在所需前一步输出文件存在的前提下)可独立执行。

其他具体参数含义可参考dpgen 官网：

另外，还需准备两个ABACUS 输入文件INPUT_rIx和INPUT_md。

INPUT_rlx主要用于优化所下载的结构，如下：

1 INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix ABACUS   
4 calculation cell-relax   
5 #Parameters (2.Iteration)   
6 ecutwfc 100   
7 scf_thr 1e-6   
8 scf_nmax 100   
9 #Parameters (3.Basis)   
10 basis_type lcao   
11 #Parameters (4.Smearing)   
12 smearing_method gauss

13 smearing_sigma 0.002   
14 #Parameters (5.Mixing)   
15 mixing_type pulay   
16 mixing_beta 0.3   
17 cal_stress 1

INPUT_md 主要用于产生一定数量的 AIMD 轨迹，用于DPGEN 迭代的输出数据，如下：

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix ABACUS   
calculation md   
5 #Parameters (2.Iteration)   
6 ecutwfc 100   
scf_thr 1e-6   
8 scf_nmax 100   
9 #Parameters (3.Basis)   
10 basis_type lcao   
11 #Parameters (4.Smearing)   
12 smearing_methodgauss   
13 smearing_sigma  0.002   
14 #Parameters (5.Mixing)   
15 mixing_type pulay   
16 mixing_beta 0.3   
17 cal_stress 1   
18 #MD   
19 chg_extrap second-order   
20 md_type 2   
21 md_tfirst 300   
22 md_tlast 300   
23 md_nstep 10   
24 md_tfirst 10   
25 md_tfreq 0.5

注：ABACUS 的MD关键字在此3.2.0新版本有所更改，如果是使用的最新版本，请严格参考手册设置关键字，如 md_type在新版本里改为了nvt、npt等形式，而不再用数字（如：2）等：

此外，还需准备一个存放布里渊区k点的文件，名为KPT:

1 K_POINTS   
3 gamma   
4444000

也可在INPUT文件里直接指定kspacing参数（注意该参数可选1维或者3维，如果是表面系统可选3维因为真空方向不需要太多k点)。

用户可自行根据所研究体系，设置、更改上述参数。

所有以上输入文件准备完毕后，采用如下命令分别对3C和2H结构运行dpgen:

nohup dpgen init_bulk param.json machine.json 1>log 2>err&

上述命令采用后台提交dpgen，等所有计算完成之后，dpgen 会自动收集训练DP势所需的数据，且转换为DeePMD-kit 的数据格式，保存在./STRU.01x01x01/02.md/sys-0036-0036/deepmd 目录里。

## 进行数据采样的迭代

run主流程包含一系列迭代，各迭代会根据所选系统在给定的温度、压力或体积等条件下进行DPMD采样与筛选，实现同步学习，所有迭代希望能够有效覆盖训练可靠模型所需的势能面上样本空间。每个迭代依次：基于init 准备的初始训练数据与之前迭代累积的训练数据训练多个新的模型->使用当前的模型进行 DPMD 采样，根据模型间预测偏差(model-deviation)挑选候选 snapshot 构象->将本轮候选 snapshot 构象进行第一性原理标定并加入数据集扩展数据集。

以上每个迭代中包含的3个步骤分别被命名为00.train，01.model_devi，02.fp，细节如下：

·00.train:使用init准备的初始训练数据与之前迭代累积的训练数据，调用DepMD-kit训练多个（默认4个）模型。模型间的唯一区别来自于初始化神经网络时使用不同的随机数种子。

. 01.model_devi:指代 model-deviation。调用LAMMPS 使用O0.train的1个模型进行MD模拟。对于任一MD中 snapshot，模型间预测偏差越大意味着当前模型系统对该 snapshot构象的精度越低，通过引人模型偏差作为误差判断并设定上下限，挑选出有希望有效改进模型对PES 整体预测精度的 snapshot构象，作为准备加入训练数据的候选构象。

·02.fp:调用ABACUS对O1.model_devi选取的候选构象进行第一性原理定标(单点计算)，并调用dpdata收集整理所得数据加入到训练数据集中。

run 执行后，相关文件即存储在...../iter.\*(迭代序号)/步骤文件夹内。以上每个迭代中的每个步骤在实现中被分解为3个阶段。以O0.train为例,3阶段分别对应:0:make_train，为训练任务准备脚本；1:run_train，依据机器配置上传文件并执行训练任务；2:post_train，收集整理分析训练任务的结果。01.model_devi与02.fp的实现步骤与之类似，分别使用3，4，5及6，7，8来标记其阶段。

用输出文件record.dpgen来记录进程已经执行过当前正在执行的阶段。如进程中断后需要续算，DP-GEN将根据此文件的记录来自前恢复run主进程。用户也可根据需求手动修改 record.dpgen 中的记录，来控制接下来将执行哪一阶段的任务：例如，删除最后一阶段(可能未执行完成即中断)的记录，并从前一个已完成阶段后恢复任务。可以通过查看record.dpgen 文件监控当前任务的执行阶段。

## 本步骤所需文件有：

。 STRU

。 \*.orb

。 \*.upf

machine.json （可以直接用 dpgen init bulk里的）

。 param.json

。 dpgen init bulk产生的初始训练数据

## 主要步骤如下：

1．新建一个文件夹，命名为：run，并进入此目录里

2.在run 文件夹里面新建init.conf 和init_data文件夹，用于存放 dpgen init bulk产生的构型和初始训练数据

3．为了区分不同构型和初始训练数据，在init.conf 和init_data文件夹里新建名为3C和 2H的两个文件夹

4.拷贝dpgen init bulk 产生的构型和初始训练数据至init.conf 和init_data 文件夹里，如:

1 cp -rf yourPath/init/3C/STRU.01x01x01/02.md/sys-0032-0032 yourPath/run/init_conf/3C

5 cp -rf yourPath/init/2H/STRU.01x01x01/02.md/sys-0036-0036/deepmd/\* yourPath/run/init_data/2H

拷贝 machine.json、\*.orb 和\*.upf 文件到 run 文件夹新建一个param.json 文件，如下案例：

{   
"type_map": ["Si","C"],   
"mass_map": [28.085,12.011],   
"init_data_prefix": "./init_data",   
"init_data_sys": [   
"3C",   
" 2H"   
],   
"sys_format":"abacus/stru",   
10 "sys_configs_prefix": "./init_conf",   
11 "sys_configs": [   
12 ["3C/sys-0032-0032/scale-1.000/00\*\*\*\*/STRU"],   
13 ["2H/sys-0036-0036/scale-1.000/00\*\*\*\*/STRU"]   
14 ],   
15 "_comment": " that's all ",   
16 "numb_models": 4,   
17 "default_training_param": {   
18 "model": {   
19 "type_map": ["Si","C"],   
20 "descriptor": {   
21 "type": "se_e2_a",   
22 "sel": "auto",   
23 "rcut_smth": 0.5,   
24 "rcut": 6.0,   
25 "neuron": [25,50，100],   
26 "resnet_dt": true,   
27 "axis_neuron": 12,   
28 "seed": 1   
29 }，   
30 "fitting_net": {   
31 "neuron": [240，240，240]，   
32 "resnet_dt": false,   
33 "seed": 1   
34 }   
35 }，   
36 "learning_rate": {   
37 "type": "exp",   
38 "start_lr": 0.001,   
39 "stop_lr": 3.5e-8,   
40 "decay_steps": 2000   
41 }，   
42 "loss": {   
43 "start_pref_e": 0.02,

44 "limit_pref_e": 2,   
45 "start_pref_f": 1000,   
46 "limit_pref_f": 1,   
47 "start_pref_v": 0.0,   
48 "limit_pref_v": 0.0   
49 }，   
50 "training": {   
51 "stop_batch": 400000,   
52 "disp_file": "lcurve.out",   
53 "disp_freq": 100,   
54 "numb_test": 10,   
55 "save_freq": 1000,   
56 "save_ckpt": "model.ckpt",   
57 "disp_training": true,   
58 "time_training": true,   
59 "profiling": false,   
60 "profiling_file": "timeline.json",   
61 "_comment": "that's all"   
62 }   
63 }，   
64 "model_devi_dt": 0.002,   
65 "model_devi_skip": 1000,   
66 "model_devi_f_trust_lo": 0.05,   
67 "model_devi_f_trust_hi": 0.10,   
68 "model_devi_clean_traj": true,   
69 "model_devi_merge_traj": true,   
70 "model_devi_jobs":[   
71 {"sys_idx": [0,1]， "temps":[50， 250，450,650]， "press": [0，100，1000,5000，1000，20000，   
72 50000]，"trj_freq": 10, "nsteps": 3000， "ensemble": "npt-tri","_idx": "01"},   
73 {"sys_idx": [0,1]， "temps":[50，250,450，650]，"press": [0，100,1000,5000，10000,20000,   
74 50000]，"trj_freq": 10, "nsteps": 3000， "ensemble": "npt-tri"，"_idx": "02"}   
75 ],   
76 "fp_style": "abacus",   
"shuffle_poscar": false,   
78 "fp_task_max": 100,   
79 "fp_task_min": 5,   
80 "fp_accurate_threshold": 0.999,   
81 "fp_accurate_soft_threshold": 0.998,   
82 "fp_Pp_path": "./",   
83 "fp_PP_files": ["Si_ONCV_PBE-1.0.upf","C_ONCV_PBE-1.0.upf"],   
84 "fp_orb_files": ["Si_gga_9au_100Ry_3s3p2d.0rb","C_gga_9au_100Ry_3s3p2d.orb"],   
85 "user_fp_params":{   
86 "ntype": 2,   
87 "symmetry": 0,   
88 "basis_type": "lcao",   
89 "ecutwfc": 100,   
90 "scf_thr": 1e-6,   
91 "scf_nmax": 100,

92 "smearing_method": "gauss",   
93 "smearing_sigma": 0.002,   
94 "mixing_type": "pulay",   
95 "mixing_beta": 0.3,   
96 "cal_force": 1,   
97 "cal_stress": 1,   
98 "kspacing": 0.08   
99 }   
100 }

·type_map：用列表方式设置了体系元素，注意顺序要和 STRU 结构文件里一致

。 mass_map：记录了对应的原子质量，和type_map 对应

oinit_data_prefix：指定初始训练集的位置所在目录，如：./init_data

·init_data_sys：定位到初始训练集的具体目录，用列表调用多个，即：./init_data/2H

。 sys_format：指定为ABACUS 计算的结构，即：abacus/stru

·sys_configs_prefix：指定探索步所需调用的结构目录，如：./init_config

·sys_configs:指定具体的某个构型的目录,用列表调用多个,即:./init_config/3c/sys-0032-0032/scale-1.0/0\*\*\*其中00 为通配符，dpgen 会遍历OO 所有文件夹的 STRU 文件

·model_devi_f_trust_lo和model_devi_f_trust_hi：可由"init数据”中力绝对值的平均值，然后取该值的20%-50%，设置成 trust level的上下限。可通过以下python 脚本计算：（注意指定初始数据位置)

```python
1 import dpdata
2 import numpy as np
3 do06=dpdata.Labeledsystem("./"， "deepmd/raw")
4 forces = np.absolute(d006["forces"])
5 af = np.average(forces)
6 f_trust_lo = af*0.2
7 f_trust_hi = af*0.5
8 print(f_trust_lo,f_trust_hi)
```

如果体系温度升高后力的幅度变大，trustlevel也要增大设置

·model_devi_jobs 里面每一行代表一轮，每轮设置参数为LAMMPS 进行MD模拟的参数

fp_style 指定，使用ABACUS 进行 SCF计算

user_fp_params里面记录ABACUS做SCF所需的参数

·其他参数含义见: https://docs.deepmodeling.com/projects/dpgen/en/latest/run/index.html

1．准备好所有输入文件之后，运行命令：

为后台提交，等待迭代计算完成即可。

2.实时监测每轮的 accurate 数据，在dpgen.log 文件里有记录

。 1）模型精度是判断每个迭代中的训练是否收敛的重要指标。

2）掌握训练策略有利于快速迭代和收敛。

·3）经验性：一般体系在5-8轮迭代内会上升到80%以上，若没有提升，如下方法排查：

·a.检查是否是探索步已经崩溃了，收集了许多不合理的结构

b.检查 train 的loss 是否是收敛

c.dp test查看fp数据是否偏差很大，或者有不合理的情况

d.检查是否是温度设置过高而导致的结构变形严重

●e.及时调整 trust level，体系温度升高后力的幅度变大，trust level也要增大设置

## 3．一些经验

## 探索步已经崩溃的解决方案

原因：DPMD跑崩的情况一般发生在训练温度高于熔点，而这个温区的结构是从”有序”到”无序态”的，其构型易变，采样空间增大，导致训练难度增加。DPMD 跑崩表明当前势函数对这类构型描述较差，大概率是因为前期数据集中这类构型采集的较少；特别是无序态结构的原子距离变小。

策略：在熔点附近的参数设置需要平滑缓慢对待，增加"MD轨迹保存频率"；缩短"MD模拟步数”或降低"MD 步长”，增加迭代次数，其DP训练势函数稳定后再延长MD 模拟步数。

## ·ABACUS收敛性测试

对每种体系一定要先做各个参数的收敛性测试，特别是：K点、能量截断，也可测试scf_thr 降低计算量。另外，对于原势文件也一定要做性质测试，看看能否能够正确描述目标体系的性质

## ）每隔几轮做一次dp test

初始数据得到后，第一轮生成了dp 势之后，一定要检查train 步骤收敛情况和做dp test。dp test 用如下命令：

dp test -m frozen_model.pb -s ./collect_data -n 20oo -d out.out

frozen_model.pb 为生成的4个模型中的某一个，2000 为collect_data 里面单个结构的总数。out.out 会生成out.e.out out.f.out out.f.out 三个偏差数据文件，用于作图判断模型优劣屏幕会输出每个构型力、能量等误差平均值，也可用\*.out文件做出DP和DFT偏差图，如：

<!-- image-->

<!-- image-->

经验性的，力的精度在1e-2量级，能量精度在5e-3量级，说明模型比较好。

## 5.6.4计算材料EOS和弹性常数的流程

本例子里提供了一个auto-test文件夹，进人之后可以结合DPGEN的auto-test命令计算材料 EOS、弹性常数的流程

auto-test的详细流程，可以从如下网页教程查阅：  
https://bohrium-doc.dp.tech/docs/software/DP-GEN_autotest  
这里贴上如何使用ABACUS 来计算材料EOS、弹性常数的相关输入文件  
relaxation.json :

{   
2 "structures":["conf"],   
3 "interaction": {   
4 "type": "abacus",   
5 "incar": " ./INPUT",   
6 "potcar_prefix":"./",   
"potcars": {"Si": "Si_ONCV_PBE-1.0.upf","C": "C_ONCV_PBE-1.0.upf"},   
8 "orb_files": {"Si":"Si_gga_9au_100Ry_3s3p2d.orb","C":"C_gga_9au_100Ry_3s3p2d.orb"}   
9 }，   
10 "relaxation": {   
11 "cal_type": "relaxation",   
12 "cal_setting":{

---

## 第 221–240 页

> ⚠️ 此切片解析失败

---

## 第 241–260 页

<table><tr><td>13</td><td> gamma_only</td><td>o</td><td></td></tr><tr><td>14</td><td> symmetry</td><td>-1</td><td># Turn off symmetry</td></tr><tr><td>15</td><td></td><td></td><td></td></tr><tr><td>16</td><td> nspin</td><td>1</td><td></td></tr><tr><td>17</td><td></td><td></td><td></td></tr><tr><td>18</td><td>out_wfc_lcao</td><td>1</td><td></td></tr><tr><td></td><td>19 out_chg</td><td>1</td><td></td></tr></table>

这里，重要的是需要设置参数out_wfc_lcao，此时你将在OUT.ABACUS 文件夹下得到各个k点的波函数系数文件，名称形如WFC_NAO_K\*.txt，这些文件将在下一步计算中被读取，并计算能带分解电荷密度。

这里的out_chg控制的是是否输出体系真实的总电荷密度，包含了所有k点、所有占据能带的贡献。因此如果你不需要总电荷密度，大可将out_chg 设为0。

## 计算并输出能带分解电荷密度

此时，我们最好再单独开一个文件夹，用来计算能带分解电荷密度。KPT和 STRU 和第1步保持一致，但是INPUT文件需要将计算类型calculation 从scf 更改为get_pchg。一个可能的 INPUT文件如下：

```shell
INPUT_PARAMETERS
2 nbands 20
3 stru_file STRU
4 kpoint_file KPT
5 pseudo_dir path/to/your/pseudo_potential/folder
6 orbital_dir path/to/your/orbital/folder
7
8 ecutwfc 100.0 # Rydberg
9
10 basis_type lcao
11
12 calculation get_pchg
13 gamma_only 0
14 symmetry -1 # Turn off symmetry
15 read_file_dir :/
16
17 # nbands_istate 2 # pchg above/below Fermi band
18 bands_to_print 2*1 2*0 1
19 if_separate_k true
20
21 nspin 1
```

我们还要将上一步得到的波函数系数文件复制到该文件夹中，推荐使用ln-s命令进行软链接。

其他参数和平面波时基本一致，仍然可以使用bands_to_print 指定需要计算的能带（见平面波计算部分）。不过在LCAO情况下，你也可以选择使用参数nbands_istate，它会选择计算Fermi面上下nbands_istate 条能带的分解电荷密度。当已经设置了bands_to_print 的情况下，nbands_istate 的效果将会被覆盖。

这里选择将if_separate_k设置为 true，表明将不同k点的贡献分开输出。

## 波函数

波函数的计算流程和电荷密度并无二异，同理第一步仍然是SCF计算，从而得到各个k点的波函数系数文件。

第二步中，我们仅需将之前的计算类型calculation 参数从get_pchg 相应地改为get_wf。这样，你就可以在OUT.ABACUS下得到波函数的模文件。如果此时你还打开了out_wfc_norm=1，还可以在OUT.ABACUS 下看到文件夹wfc_realspace，里面存放了（不同能带、不同k点、不同自旋的）波函数的实部文件和虚部文件。

## 6.3.6示例：对电荷密度进行可视化

以单层石墨烯为例，其原胞的 STRU 文件如下：

```csv
1 ATOMIC_SPECIES
2C 12.011 C_ONCV_PBE-1.0.upf
3
4 NUMERICAL_ORBITAL
5 6_C_TZDP/C_gga_10au_100Ry_3s3p2d.orb
6
7 LATTICE_CONSTANT
8 1.889726
9
10 LATTICE_VECTORS
11 2.4600000000 0.0000000000 0.0000000000
12 -1.230000000 2.1304220000 0.0000000000
13 0.0000000000 0.0000000000 10.0000000000
14
15 ATOMIC_POSITIONS
16 Direct
17
18
19 0.0000000000
20 2
21 0.000000000 0.0000000000 0.5000000000 00 0 mag 0.0
22 0.333333333 0.6666666666 0.5000000000 0 0 0 mag 0.0
```

我们选择围绕Γ点2×2x1的k点采样（仅作演示用，并不是对该体系收敛的最优采法)，KPT文件如下：

```csv
1 K_POINTS
2 0
3 Gamma
4221000
```

SCF 计算中可以得到输出的kpoints 文件，指示了我们输出文件中k点编号和倒空间（分数）坐标的对应关系：

```python
1 nkstot now = 4
2 K-POINTS DIRECT COORDINATES
3 KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT
4 1 0.00000000 0.00000000 0.00000000 0.2500
5 2 0.50000000 0.00000000 0.00000000 0.2500
6 3 0.00000000 0.50000000 0.00000000 0.2500
7 4 0.50000000 0.50000000 0.00000000 0.2500
```

倒空间基矢的选择可以在输出日志running_scf.log 中看到，形如：

<table><tr><td colspan="4">1 Reciprocal vectors: (Cartesian coordinate: in unit of 2 pi/a_0)</td></tr><tr><td>2</td><td>+0.406504</td><td>+0.234695</td><td>10</td></tr><tr><td>3</td><td>10</td><td>+0.469391</td><td>10</td></tr><tr><td>4</td><td>+0</td><td>!</td><td>+0.1</td></tr></table>

采用 SeeK-path工具中可以看到Brillouin区的样子：

<!-- image-->

这里 SeeK-path 是一个能够自动生成标准化的k 点路径的工具，方便用于各种电子结构计算，可通过网址https://www.materialscloud.org/work/tools/seekpath 访问。

不难发现，实际上2×2x1的Brillouin区采样只能采到Γ点和M点（编号2、3、4的k点都等价为M点，在symmetry=1时会显式约化)。可以预想到，K2、K3、K4在实空间中的分布形状应该是一致的，它们只相差相应的晶格对称性操作。

以LCAO 为例,开启共线磁矩nspin=2(仅作为演示,没有磁性的石墨烯自旋上下结果应该完全一致),在第一步 SCF后可以在OUT.ABACUS 文件夹下得到如下波函数系数文件:WFC_NAO_K1.txt、…、WFC_NAO_K8.txt。这里K1到K8代表8个k点，k点编号比实际设置时多了一倍是因为其中编号1\~4是代表自旋向上的波函数文件，而编号5～8是代表自旋向下的波函数文件。

SCF 计算结束后，将得到的这些系数文件复制/链接到另一个我们自己创建的新文件夹下，根据上面的教程设置控制文件，开启计算类型calculation=get_pchg 的新计算，就可以在OUT.ABACUS 文件夹下得到如下的电荷密度文件。若选择if_separate_k=true，输出文件将形如：

1 BAND1_K1_SPIN1_CHG.cube   
2 BAND1_K1_SPIN2_CHG.cube   
3 BAND1_K2_SPIN1_CHG.cube   
4 BAND1_K2_SPIN2_CHG.cube   
5

若选择if_separate_k=false，则输出文件名中不会带有k点信息。这些cube 格式文件可以直接在VESTA软件中打开并可视化。

例如，第一条能带「点自旋向上的贡献存储在BAND1_K1_SPIN1_CHG.cube文件里，可视化效果为：

<!-- image-->

第二条能带三个不同 M 点的贡献为 BAND2_K2_SPIN\*_CHG.cube、BAND2_K3_SPIN\*_CHG.cube、BAND2_K4_SPIN\*_CHG.cube :

<!-- image-->

<!-- image-->

<!-- image-->

可以发现，它们确实形状一致，但是在实空间被旋转了不同的角度，符合预期。

## 6.3.7结语

以上就是本教程的内容，希望对读者有所帮助。若有问题，请在 ABACUS GitHub issue 中向我们反馈，或者发送邮件联系本教程作者。感谢大家对ABACUS 软件的支持。

## 6.4ABACUS计算电子局域函数ELF使用教程

## 6.4.1背景

电子局域函数（Electron localization function，ELF）最早在1990 年由 Becke 和Edgecombe 提出[1]，它描述电子的局域化程度。根据定义，ELF是一个0到1之间的数，越接近1，表示电子越局域，ELF=1/2时，对应均匀电子气，越接近0，表示电子越离域。

目前，ABACUS 支持使用平面波基组、原子轨道基组下的 Kohn-Sham DFT（KSDFT）和无轨道密度泛函理论计算并输出ELF。

## 6.4.2定义

## 自旋非极化

自旋非极化时，ELF定义为

$$
E L F = \frac { 1 } { 1 + \chi ^ { 2 } }
$$

其中 $\begin{array} { r } { \chi = \frac { \tau _ { K S } - \tau _ { v W } } { \tau _ { T F } } , \ \tau _ { K S } = \frac { 1 } { 2 } \sum _ { i } f _ { i } | \nabla \psi _ { i } | ^ { 2 } } \end{array}$ 为KSDFT的正定动能密度,

$$
\tau _ { v W } = \frac { | \nabla \rho | ^ { 2 } } { 8 \rho }
$$

为 von Weizsäcker 泛函的正定形式, $\begin{array} { r } { \tau _ { T F } = \frac { 3 } { 1 0 } ( 3 \pi ^ { 2 } ) ^ { 2 } / 3 \rho ^ { 5 / 3 } } \end{array}$ 为 Thomas-Fermi 泛函。

## 自旋极化

自旋极化状态下，我们可以针对一种自旋的电子定义ELF，即

$$
\mathrm { E L F } _ { \sigma } = \frac { 1 } { 1 + \chi _ { \sigma } ^ { 2 } }
$$

其中 $\begin{array} { r } { \chi _ { \sigma } = \frac { \tau _ { \sigma } ^ { K S } - \tau _ { \sigma } ^ { v } W } { \tau _ { \sigma } ^ { T F } } , \tau _ { \sigma } ^ { K S } = \frac { 1 } { 2 } \sum _ { i } f _ { i } | \nabla \psi _ { i , \sigma } | ^ { 2 } } \end{array}$ 为自旋为σ的电子对应的正定动能密度。对于动能泛函，根据自旋标度率，有 $\begin{array} { r } { T _ { \sigma } = \frac { 1 } { 2 } T [ 2 \rho _ { \sigma } ] } \end{array}$ ，因此， $\begin{array} { r } { \tau _ { \sigma } ^ { v } W = \frac { | \nabla \rho _ { \sigma } | ^ { 2 } } { 8 \rho _ { \sigma } } , \ \tau _ { \sigma } ^ { T F } = \frac { 1 } { 2 } \frac { 3 } { 1 0 } ( 3 \pi ^ { 2 } ) ^ { 2 / 3 } ( 2 \rho _ { \sigma } ) ^ { 5 / 3 } = \frac { 3 } { 1 0 } ( 6 \pi ^ { 2 } ) ^ { 2 / 3 } \rho _ { \sigma } ^ { 5 / 3 } } \end{array}$ 。

注：Becke 和 Edgecombe对ELF的原始定义为

$$
\mathrm { E L F } = \left[ 1 + \left( \frac { \tau _ { \sigma } - \frac { ( \nabla \rho _ { \sigma } ) ^ { 2 } } { 4 \rho _ { \sigma } } } { \frac { 3 } { 5 } ( 6 \pi ^ { 2 } ) ^ { 2 / 3 } ( \rho _ { \sigma } ^ { 5 / 3 } ) } \right) ^ { 2 } \right] ^ { - 1 }
$$

由于这里 $\begin{array} { r } { \tau _ { \sigma } = \sum _ { i } f _ { i } | \nabla \psi _ { i , \sigma } | ^ { 2 } = 2 \tau _ { \sigma } ^ { K S } } \end{array}$ ，因此分子分母同除以2后与上述定义相同。

同时，我们也可以定义总的ELF，即

$$
\mathrm { E L F } _ { t o t } = \frac { 1 } { 1 + \chi _ { t o t } ^ { 2 } }
$$

其中 $\begin{array} { r } { \chi _ { t o t } = \frac { \tau _ { \alpha } ^ { K S } + \tau _ { \beta } ^ { K S } - \tau _ { \alpha } ^ { v } W - \tau _ { \beta } ^ { v } W } { \tau _ { \alpha } ^ { T F } + \tau _ { \beta } ^ { T F } } . } \end{array}$

## 稳定性修正

在远离原子核的区域，×的分子和分母均趋于零，但分母更快地趋于零，因此×趋于无穷，最终ELF趋于零。

但在LCAO基组下，由于基组的不完备性，在远离原子核的某些区域，×的分子可能更快地趋于零，导致X趋于零或得到一个有限的值，最终得到非零的ELF，即电子出现不物理的”局域化”。在PW基组下，则没有上述问题。

为了修正基组带来的不稳定性，我们采用了文献[2]中的建议，在x的分子上增加一个很小的修正值，以保证×在远离原子核的区域正确地趋于零。经测试后，ABACUS取此修正值为10-5，即可以提升稳定性，又不会影响近核区域的结果。

## 6.4.3ABACUS计算ELF具体流程

相关参数：out_elf[int]［int]：out_elf 的第一个值控制是否输出 ELF，设置为1则输出，默认值为0；第二个值控制输出的有效数字位数，默认为3。

在INPUT 中设置out_elf 1并运行 ABACUS 后，ELF会被存储到OUT文件夹下。

自旋非极化情况下（nspin1）ELF会被存储到 ELF.cube文件中；自旋极化的情况下（nspin 2），除了ELF.cube中存储的总ELF，自旋为上和自旋为下的ELF会分别存储在ELF_SPIN1.cube和ELF_SPIN2.cube中。这些cube 文件都可以直接在VESTA中可视化。

下面举两个具体的例子，一个自旋非极化的水分子，另一个是自旋极化的体心立方铁。

## 计算实例：自旋非极化的水分子

算例地址:https://github.com/MCresearch/abacus-user-guide/tree/master/examples/elf/H20-pw

运行此算例得到的ELF.cube 文件如下，由于此算例中的水分子放置在真空中，远离水分子的区域内 ELF都为零。

```csv
1 STEP: O Cubefile created from ABACuS. Inner loop is z, followed by y and x
2 1 (nspin)
3 3 0.00.0 0.0
4 180 0.155556 0.000000 0.000000
5180 0.000000 0.155556 0.000000
6 180 0.000000 0.000000 0.155556
71 1.000000 11.967714 18.726812 8.322864
8 1 1.000000 14.244302 20.657387 7.503683
9 8 6.000000 13.550429 19.681072 9.139011
10 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
11 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
12 ····
13 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00
```

将此文件在VEASTA中可视化，得到下图，其中左图为ELF的等高面，右图为在H-O-H截面上的切面图。

<!-- image-->

<!-- image-->

## 原子轨道基组

算例地址:https:/github.com/MCresearch/abacus-user-guide/tree/master/examples/elf/H20-lcao

运行此算例后，可得到与上例格式相同的ELF.cube文件，用VESTA可视化的结果如下图，其中左图为ELF的等高面，右图为在H-O-H截面上的切面图。

<!-- image-->

<!-- image-->

由于使用的基组不同，得到的ELF也略有不同。

## 6.4.4计算实例：自旋极化的体心立方铁

算例地址:https:/github.com/MCresearch/abacus-user-guide/tree/master/examples/elf/bcc-Fe-pw

此算例使用PW基组下的KSDFT。运行此算例后，可以得到ELF.cube，ELF_SPIN1.cube与ELF_SPIN2.cube三个文件，分别存储总ELF，自旋为上电子的ELF以及自旋为下电子的ELF，下面为三种不同的ELF在(100)晶面上的分布图。

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

## 6.4.5参考文献

[1]Becke A D,Edgecombe KE. A simple measure of electron localization in atomic and molecular systems[J]. The Journal of chemical physics,1990, 92(9): 5397-5403.

[2]卢天,陈飞武.电子定域化函数的含义与函数形式[J].物理化学学报,2011,27(12):2786-2792.

## 6.5ABACUS+Badercharge 分析教程

## 6.5.1前言

我们在软件开发时不可避免遇到化学相关应用场景的需求，其中一个可能重点关注的信息便是“价态”。尽管从原理上考虑，“价态”这个量的定义并不明确，但化学场景的使用者总希望找到各式各样的方法来契合他们的“化学直觉”，便诞生了各种布居分析（Population Analysis）方法：如Mullken、Hirshfeld、Hirshfeld-Iterative(Hirshfeld-I)、ADCH、DDEC6、CM5、Mayer、Bader等，以及各种电子的定域化分析方法，如Wannier、ELF等。尽管他们各自存在或致命或物理上定义不明的问题，如Mullken 电荷具有非常显著的基组相关效应，Hirshfeld电荷经常对过渡元素计算出携带负电，Bader对于电荷分布极其不均匀的体系会得出异常结果等，但仍然有足够普遍的受众群体。

Bader charge 是将电子密度进行类似于 Voronoi tesellation的剖分，在原子间以电荷密度的梯度为0处创建截面，然后将原子周围所有截面包络出的cell进行实空间积分，即为Bader charge。考虑实际计算，我们总能对价电子的密度进行切分，但对近核电子的处理则有所差别：CP2K支持构造出核电荷分布，VASP支持从PAW势中重建核电荷分布，而QE和ABACUS 则不支持以上处理方式。

## 6.5.2Bader分析软件部署

网站：https:/theory.cm.utexas.edu/henkelman/code/bader/

我们可以直接运行linux命令以下载 Bader电荷分析软件的可执行文件版本，也可以选择下载源码然后手动编译:

```shell
1 vget https://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz
2 tar -zxvf bader_lnx_64.tar.gz
3 chmod +x bader
```

下载源码:

1 wget https://theory.cm.utexas.edu/henkelman/code/bader/download/bader.tar.gz   
2 tar -zxvf bader.tar.gz   
3 cd bader   
4 make   
5 chmod +x bader

## 6.5.3使用示例

在了解bader charge 的原理后，我们首先从 nspin1（only one spin channel）开始讲解具体使用。

## nspin 1 case

生成所需cube 文件

bader程序需要读取以cube方式，或者CHGCAR 格式存储的实空间电荷密度，因此运行 ABACUS 时，只需要添加

1 out_chg 1

即可输出电荷密度。

只需要以

```batch
1 /path/to/bader Si2-ELECTRON_DENSITY.cube
```

即可，屏幕会有如下输出：

1 GRID BASED BADER ANALYSIS (VerSiOn 1.05 08/19/23)  
2  
3 OPEN ... Si2-ELECTRO  
GAUSSIAN-STYLE INPUT FILE  
DENSITY-GRID: 60 x 60 x 60  
6 CLOSE ... Si2-Si2.cube-ELECTRO  
RUN TIME: 0.02 SECONDS  
8  
9 CALCULATING BADER CHARGE DISTRIBUTION  
10 0　10-25　-50-75-100  
11 PERCENT DONE: \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
12  
13 REFINING AUTOMATICALLY  
14 ITERATION: 1  
15 EDGE POINTS : 97653  
16 REASSIGNED POINTS: 7908  
17  
18 RUN TIME: 0.23 SECONDS  
19  
20 CALCULATING MINIMUM DISTANCES TO ATOMS  
21 010　25　50-75　100  
22 PERCENT DONE: \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

同时生成了ACF.dat，AVF.dat和BCF.dat 文件，ACF.dat 中即是bader 切分该电荷密度后在“Voronoi cell”切分的结果：

<table><tr><td>1 #</td><td></td><td>X</td><td>Y</td><td>Z</td><td>CHARGE</td><td>MIN DIST</td><td>ATOMIC VOL</td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>1</td><td>3.653762</td><td>2.109500</td><td>1.491642</td><td></td><td>3.999891</td><td>1.952476</td><td>137.965434</td></tr><tr><td>4</td><td>2</td><td>0.000000</td><td>0.000000</td><td>0.000000</td><td>4.000124</td><td></td><td>1.952474</td><td>137.962879</td></tr><tr><td>5</td><td colspan="8"></td></tr><tr><td>6</td><td>VACUUM</td><td>CHARGE:</td><td></td><td>0.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>7</td><td></td><td>VACUUM VOLUME :</td><td></td><td>0.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td>NUMBER</td><td>OF</td><td>ELECTRONS:</td><td>8.0000</td><td></td><td></td><td></td><td></td></tr></table>

我们将“CHARGE”列数据和赝势中PP_HEADER 部分的 z_Valence 值相减，即可得到该原子所携带电荷量。

## nspin 2 case

ABACUS 目前分 spin channel 进行电荷密度的输出，若 nspin 2，则会在目录下输出 SPIN1_CHG.cube 和SPIN2_CHG.cube。然而，直接对每一个 spin channel进行体积剖分，然后按照原子序号相加的结果可能是错误的，因为两个 spinchannel中不一定每个原子均具有位置相同的“0梯度面”。因此对同一原子，可能在两个 spinchannel中具有不同大小的cell。为了避免这种情况，可以首先将两cube 文件进行加和，之后再调用 bader.X。

## Cube manipulator

Cube manipulator 是处理 Gaussian cube 格式 3D实空间格点数据的小工具，可以在 ABACUS 代码包 tools 文件夹下找到（htps:/github.com/deepmodeling/abacus-develop/blob/develop/tools/plot-tools/cube_manipulator.py）。

通过-help可以查看可用选项：

1 myaccount@mycomputer:\~/abacus-develop/examples/spin_polarized/AFM/OUT.ABACUS# python3   
2 cube_manipulator.py --help   
3 usage: cube_manipulator.py [-h] [-i INP] [-o OUT] [-s SCALE] [--p1d P1D] [--s2d S2D] [-p PLUS]   
4 [-m MINUS]   
5   
6 manipulate the Gaussian cube format volumetric data.   
7   
8 options:   
9 -h，--help show this help message and exit   
10 -i INP,--inp INP the input Gaussian cube file.   
11 -0 OUT，--out OUT the output file.   
12 -s SCALE， --scale SCALE   
13 scale the Gaussian cube file by a factor.

14 --p1d P1D integrate the Gaussian cube file in 2D, followed by the axis: 'x', ..   
15 --s2d S2D slice the Gaussian cube file along one axis, followed by string like   
16 'x=0.0'，'y=0.O'，'z=O.0'. Note: should be fractional coodinate.   
17 -p PLUS, --plus PLUS plus the two Gaussian cube files.   
18 -m MINUS, --minus MINUS   
19 minus the two Gaussian cube files.   
20   
21 Once meet any problem, please submit an issue at: https://github.com/deepmodeling/   
22 abacus-develop/issues

## 生成所需cube 文件

我们选择abacus-develop/examples/spin_polarized/AFM算例，将 INPUT文件中out_chg flag 修改为1，得到两个cube 文件分别对应于两个 spin channel。此时我们只需要将两个 SPIN\*_CHG.cube 相加即可：

```batch
1 python3 cube_manipulator.py -i SPIN1_CHG.cube -p SPIN2_CHG.cube -o ELECTRONIC_DENSITY.cube
```

运行bader电荷分析  
对文件 ELECTRONIC_DENSITY.cube 使用 bader.X，得到：

1 GRID BASED BADER ANALYSIS (Version 1.05 08/19/23)  
2  
3 OPEN ... ELECTRONIC_DENSITY.C  
4 GAUSSIAN-STYLE INPUT FILE  
5 DENSITY-GRID: 36 x 36 x 36  
6 CLOSE ... ELECTRONIC_DENSITY.C  
RUN TIME: 0.00 SECONDS  
8  
9 CALCULATING BADER CHARGE DISTRIBUTION  
10 0 10-25-50-75　100  
11 PERCENT DONE:  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
12  
13 REFINING AUTOMATICALLY  
14 ITERATION: 1  
15 EDGE POINTS: 39985  
16 REASSIGNED POINTS: 6753  
17  
18 RUN TIME: 0.07 SECONDS  
19  
20 CALCULATING MINIMUM DISTANCES TO ATOMS  
21 0 10-25-50-75　100  
22 PERCENT D0NE: \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
23 RUN TIME: 0.00 SECONDS  
24  
25 WRITING BADER ATOMIC CHARGES TO ACF.dat  
26 WRITING BADER VOLUME CHARGES TO BCF.dat  
27  
28 NUMBER OF BADER MAXIMA FOUND: 108  
29 SIGNIFICANT MAXIMA FOUND: 108

该例为，呈反铁磁态，每个Fe原子有16个价电子，最终积分得到电子数量为31.9909，因此符合预期。同样查看 ACF.dat文件：

<table><tr><td>1 #</td><td></td><td>X</td><td>y</td><td>Z</td><td>CHARGE</td><td>MIN DIST</td><td>ATOMIC VOL</td><td></td></tr><tr><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>1</td><td>0.000000</td><td>0.000000</td><td>0.000000</td><td>15.999544</td><td></td><td>2.060461</td><td>76.704205</td></tr><tr><td>4</td><td>2</td><td>2.676622</td><td>2.676622</td><td>2.676622</td><td>15 .999544</td><td></td><td>2.060459</td><td>76.704205</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td></td><td>vacuum chArgE :</td><td></td><td>0.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>7</td><td></td><td>VACUUM VOLUME :</td><td></td><td>0.0000</td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td></td><td>NUMBER OF</td><td>ELectrons:</td><td>31.9991</td><td></td><td></td><td></td><td></td></tr></table>

发现两Fe原子均为电中性，同样符合预期。

## 6.5.4 附录：cube manipulator 更多的使用方法

Cube manipulator还有除了加法之外的更多功能，例如两cube文件进行减法、一个cube文件进行乘法等，以及还有面向更实际的应用场景，2D平面积分与cube3D内容切片。

## AXPY类操作

AXPY是BIAS中函数，代表了诸如：

$$
y = \alpha x + \beta y
$$

的操作。因此-p flag 即为α=1,θ=1,-m flag 即为α=1,θ=-1,-s为α=?,θ=0。例如将 SPIN1_CHG.cube 和 SPIN2_CHG.cube 相减可以获得自旋密度：

```batch
1 python3 cube_manipulator.py -i SPIN1_CHG.cube -m SPIN2_CHG.cube -o SPIN_DENSITY.cube
```

## Profileld操作

在进行表面模型的功函数计算时，需要对Hartree 势进行面积分，得到沿某个轴的1维曲线。我们分别使用ELECTRON_DENSITY.cube 和 SPIN_DENSITY.cube 对该功能进行演示：

```batch
1 python3 cube_manipulator.py -i ELECTRON_DENSITY.cube --p1d x -o charge1d.dat
2 python3 cube_manipulator.py -i SPIN_DENSITY.cube --p1d x -o spin1d.at
```

<!-- image-->  
图1.SPIN1_CHG.cube

<!-- image-->  
图2.SPIN2_CHG.cube

<!-- image-->  
图3.SPIN_DENSITY.cube

即对yz平面进行积分，进行简单的绘图：

<!-- image-->  
图4.chargeld.dat，0和0.5位置各有一个Fe原子，Bader 电荷切分的地方接近0.25和0.75

<!-- image-->  
图5.spinld.dat，spin density 由正到负，符合反铁磁 Fe2 的预期

## Slice2d操作

有时我们会考察某个截面的电子密度，对原子周围电子的局域化程度进行分析。我们仍然分别使用ELEC-TRON_DENSITY.cube 和 SPIN_DENSITY.cube 对该功能进行演示：

```batch
python3 cube_manipulator.py -i ELECTRON_DENSITY.cube --s2d x=O -o charge2d_000.dat
2 python3 cube_manipulator.py -i ELECTRON_DENSITY.cube --s2d x=0.25 -o charge2d_025.dat
3 python3 cube_manipulator.py -i ELECTRON_DENSITY.cube --s2d x=0.5 -o charge2d_050.dat
4
5 python3 cube_manipulator.py -i SPIN_DENSITY.cube --s2d x=O -o spin2d_000.dat
6 python3 cube_manipulator.py -i SPIN_DENSITY.cube --s2d x=0.25 -o spin2d_025.dat
7 python3 cube_manipulator.py -i SPIN_DENSITY.cube --s2d x=0.5 -o spin2d_050.dat
```  
进行简单绘图，电荷密度：

<!-- image-->  
图6.charge2d_000.dat

<!-- image-->  
图7.charge2d_025.dat

<!-- image-->  
图8.charge2d_050.dat

自旋密度：  
<!-- image-->  
图9.spin2d_000.dat

<!-- image-->  
图10.spin2d_025.dat

<!-- image-->  
图11.spin2d_050.dat

## 6.5.5 Have a question? Submit issue!

如果在使用 ABACUS+Bader Analysis 过程中发现了bug或者运行结果不达预期,可以在deepmodeling/abacus-develop 仓库下提交 issue。

## 6.6 ABACUS+pymatgen 计算弹性常数

## 6.6.1介绍

本教程旨在描述如何采用ABACUS来计算材料体系的弹性常数（elastic constants）。弹性常数是表征材料弹性的量，也是材料的重要力学性质，可以通过原子尺度模拟计算获得，例如通过密度泛函理论或者原子间的势函数方法。在晶体结构预测辅助的新材料计算设计中，弹性常数常用于检查结构的稳定性。

在晶体的线弹性范围内，应力o和相应应变ε之间的关系符合胡克定律，

$$
\sigma = C \varepsilon , \sigma _ { i j } = C _ { i j k l } \varepsilon _ { k l } , \quad i , j , k , ( x , y , z ) , \quad C _ { i j k l }
$$

由于σ和ε均是对称张量，基于Voigt表示法：

$$
x x \mapsto 1 , y y \mapsto 2 , z z \mapsto 3 , y z \mapsto 4 , x z \mapsto 5 , x y \mapsto 6 ,
$$

$$
\sigma _ { 1 } = \sigma _ { x x } , \sigma _ { 2 } = \sigma _ { y y } , \sigma _ { 3 } = \sigma _ { z z } , \sigma _ { 4 } = \sigma _ { y z } , \sigma _ { 5 } = \sigma _ { x z } , \sigma _ { 6 } = \sigma _ { x y }
$$

$$
\epsilon _ { 1 } = \epsilon _ { x x } , \epsilon _ { 2 } = \epsilon _ { y y } , \epsilon _ { 3 } = \epsilon _ { z z } , \epsilon _ { 4 } = \epsilon _ { y z } , \epsilon _ { 5 } = \epsilon _ { x z } , \epsilon _ { 6 } = \epsilon _ { x y }
$$

此时，

$$
C _ { x x x x } \mapsto C _ { 1 1 } , C _ { x x y y } \mapsto C _ { 1 2 } , . . .
$$

$$
{ \left[ \begin{array} { l } { \sigma _ { 1 } } \\ { \sigma _ { 2 } } \\ { \sigma _ { 3 } } \\ { \sigma _ { 4 } } \\ { \sigma _ { 5 } } \\ { \sigma _ { 6 } } \end{array} \right] } = { \left[ \begin{array} { l l l l l l l l } { C _ { 1 1 } } & { C _ { 1 2 } } & { C _ { 1 3 } } & { C _ { 1 4 } } & { C _ { 1 5 } } & { C _ { 1 6 } } & { \epsilon _ { 1 } } \\ { C _ { 1 2 } } & { C _ { 2 2 } } & { C _ { 2 3 } } & { C _ { 2 4 } } & { C _ { 2 5 } } & { C _ { 2 6 } } & { \epsilon _ { 2 } } \\ { C _ { 1 3 } } & { C _ { 2 3 } } & { C _ { 3 3 } } & { C _ { 3 4 } } & { C _ { 3 5 } } & { C _ { 3 6 } } & { \epsilon _ { 3 } } \\ { C _ { 1 4 } } & { C _ { 2 4 } } & { C _ { 3 4 } } & { C _ { 4 4 } } & { C _ { 4 5 } } & { C _ { 4 6 } } & { 2 \epsilon _ { 4 } } \\ { C _ { 1 5 } } & { C _ { 2 5 } } & { C _ { 3 5 } } & { C _ { 4 5 } } & { C _ { 5 5 } } & { C _ { 5 6 } } & { 2 \epsilon _ { 5 } } \\ { C _ { 1 6 } } & { C _ { 2 6 } } & { C _ { 3 6 } } & { C _ { 4 6 } } & { C _ { 5 6 } } & { C _ { 6 6 } } & { 2 \epsilon _ { 6 } } \end{array} \right] }
$$

如果我们通过施加应变 $\varepsilon _ { i }$ 使晶体变形并计算相应的应力，则可以从方程中获得弹性常数。晶体晶胞上的变形矩阵为 $D = I + \varepsilon$ ，其中I是3\*3的单位矩阵，ε是Voigt表示法中的应变矩阵。

在3维晶体中，应变矩阵为 $\varepsilon = { \left[ \begin{array} { l l l } { \varepsilon _ { 1 } } & { \varepsilon _ { 6 } } & { \varepsilon _ { 5 } } \\ { \varepsilon _ { 6 } } & { \varepsilon _ { 2 } } & { \varepsilon _ { 4 } } \\ { \varepsilon _ { 5 } } & { \varepsilon _ { 4 } } & { \varepsilon _ { 3 } } \end{array} \right] }$ ，在2维平面晶体中，应变矩阵为

$$
\varepsilon = { \left[ \begin{array} { l l l } { \varepsilon _ { 1 } } & { \varepsilon _ { 6 } } & { 0 } \\ { \varepsilon _ { 6 } } & { \varepsilon _ { 2 } } & { 0 } \\ { 0 } & { 0 } & { 0 } \end{array} \right] }
$$

变形后，晶格矢量为 $A ^ { \prime } = A \cdot D$ ，其中A是初始的晶格矢量。

在实际计算中，6种应变状态将逐一应用于初始驰豫（relaxation）后的结构，因此每次只考虑一种独立变形。对于6种应变状态中的每一种，应用4种不同的默认应变大小：

$$
\varepsilon _ { i } \in \{ - 0 . 0 1 , - 0 . 0 0 5 , 0 . 0 0 5 , 0 . 0 1 \} ,
$$

将产生24种构型。接着对这24种构型分别使用DFT计算（固定晶格矢量，但允许原子驰豫，即relax），对所获得的每种应变状态的4个应力和应变的集合使用线性拟合来计算相应的弹性常数。

更多细节可以参考：Elastic Constants 丨Materials Project Documentation

## 6.6.2准备

## 下载例子

首先可以下载一个ABACUS 的计算实例，可以从Gitee上下载。具体来说，可以通过在网页右侧点击克隆/下载->下载ZIP得到算例，或者在linux 终端执行如下命令得到算例：

下载后解压，之后进入abacus-user-guide/examples/elastic 文件夹

里面有relax文件夹，为8原子金刚石的算例，周期性边界条件。

此外，gene_dfm.py 和 compute_dfm.py 为计算弹性常数使用的 python 脚本。

run_task.sh 和 sub.sh 为批量运行 abacus 计算的 shell 脚本，INPUT、KPT、C_ONCV_PBE-1.0.upf 和C_gga_7au_100Ry_2s2pld.orb为运行 abacus 计算所需的输入文件。

## 下载并安装 pymatgen

Python Materials Genomics (pymatgen，https:/pymatgen.org/）是一个API包，该软件可以与 materials project结合进行高通量计算。该软件包是由加州大学圣地亚哥雅各布斯工程学院的纳米工程教授 Shyue Ping Ong 和他的材料虚拟实验室(Materials VirtualLab)团队开发并维护的程序。

在本教程里会用到 pymatgen来计算弹性常数，此外还会用到dpdata，monty，numpy这三个库，可以使用如下命令安装：

1 pip install monty numpy dpdata pymatgen

弹性常数所用方法相关文档：https://ymatgen.org/pymatgen.analysis.elasticity.html

## 6.6.3 金刚石的弹性常数计算

## 结构弛豫

进入relax文件夹，运行ABACUS 进行结构弛豫，完成后在OUT.C8文件夹下出现 STRU_ION_D文件。之后的应变都将在这个构型文件基础上产生。此外，计算弹性常数所用的应力也会减去这个构型的应力，因此也需要running_relax.log 输出应力

## 产生应变构型

回到例子根目录，执行如下命令：

1 python gene_dfm.py abacus

默认应变的大小（代表晶格常数的倍数，例如0.01，其形变量是1

```hcl
1 norm_strains = [-0.010,-0.005,0.005,0.010]
2 shear_strains = [-0.010,-0.005,0.005,0.010]
```

将会产生task.000到task.023共24个文件夹，分别对应第一节所说的24种构型。进人任意task文件夹，其中会有INPUT、KPT、STRU 和 strain.json 四个文件。

其中 INPUT 和 KPT拷贝自例子根目录下的 INPUT 和 KPT，由于使用的赝势和轨道文件放在例子根目录下，所以pseudo_dir和orbital_dir均设置为./。此外需要进行固定晶格矢量，但允许原子弛豫来计算应力，因此calculation 设置为 relax。STRU 和 strain.json 分别是生成的构型文件和相应的应变大小。

## 计算应力

分别进入上述task文件夹，运行abacus 计算相应构型的应力。也可以使用例子根目录下的run_task.sh脚本，但要注意根据实际修改其和 sub.sh 的内容。

## 计算弹性常数

回到例子根目录，执行如下命令：

1 python compute_dfm.py abacus

屏幕输出如下:

---

## 第 261–280 页

```python
8 # Bulk  Modulus BV = 419.37 GPa
9 # Shear Modulus GV = 521.42 GPa
10 # Youngs Modulus EV = 1105.91 GPa
11 # Poission Ratio uV = 0.06
```

第2到7行即为计算的各种弹性常数（单位：GPa)，分别是体弹性模量（Bulk Modulus）、剪切模量（ShearModulus）、杨氏模量（Youngs Modulus）和泊松比（Poission Ratio）。精度更高的计算结果存储在 elastic.json 内。以下是常用的几个弹性常数相关的名词解释：

弹性模量（BulkModulus)：指当有力施加于物体或物质时，其弹性变形（非永久变形）趋势的数学描述。物体的弹性模量定义为弹性变形区的应力-应变曲线的斜率。

杨氏模量（Young's Modulus)：是材料力学中的名词，一般将杨氏模量习惯称为弹性模量。弹性材料承受正向应力时会产生正向应变，在形变量没有超过对应材料的一定弹性限度时，定义正向应力与正向应变的比值为杨氏模量。杨氏模量的大小标志了材料的刚性，杨氏模量越大，越不容易发生形变。大部分金属在合金成分不同、热处理在加工过程中的应用，其杨氏模量值会有5%或更大的波动。

剪切模量（Shear Modulus)：又称刚度模量，描述材料在剪切力作用下抵抗形变的能力，即物体形变的大小与作用的剪切应力之间的关系。

泊松比(Poisson's Ratio)：当材料在一个方向被压缩，它会在与该方向垂直的另外两个方向伸长，这就是泊松现象，泊松比是用来反映泊松现象的无量纲的物理量。泊松比一般是正值，表示在一方向拉伸后，在其他方向收缩。不过也存在泊松比为零（在一方向拉伸后，在其他方向的尺寸不变)，其至为负的材料。

## 6.7 ABACUS+Phonopy 计算声子谱

## 6.7.1介绍

本教程旨在介绍采用ABACUS（基于 ABACUS 3.2.2版本）做密度泛函理论计算，并且结合Phonopy 软件计算声子谱的流程。此外，本教程还用到gnuplot来绘图。

首先推荐大家阅读以下文档中的详细说明：

ABACUS 官方文档：Phonopy-ABACUS documentation

Phonopy 相关文档：ABACUS & phonopy calculation—Phonopy v.2.19.1

Gnuplot 主页：gnuplot homepage

## 6.7.2准备

我们以FCC Al这个简单例子来演示使用有限位移方法来结合 ABACUS 和Phonopy 计算声子谱。

## 下载并安装 Phonopy

1 git clone https://github.com/phonopy/phonopy.git   
2 cd phonopy   
3 python3 setup.py install

## 下载FCC Al例子

可以从 Gitee上下载。可以在网页右侧点击克隆/下载->下载 ZIP 得到算例，或者在linux 终端执行如下命令得到算例：

1 git clone https://gitee.com/mcresearch/abacus-user-guide.git

下载后解压，之后进入abacus-user-guide/examples/interface_Phonopy 文件夹

## 6.7.3流程

## 使用ABACUS优化结构

这里我们已经给出一个已经优化好的FCCAl结构ABACUS中的结构文件名为STRU:

1 ATOMIC_SPECIES   
2 Al 26.982 Al_0NCV_PBE-1.0.upf upf201   
3   
4 NUMERICAL_ORBITAL   
5 Al_gga_7au_100Ry_4s4p1d.orb   
6   
7 LATTICE_CONSTANT   
8 1.88972612546   
9   
10 LATTICE_VECTORS   
11 4.03459549706 0 0 #latvec1   
12 0 4.03459549706 0 #latvec2   
13 0 0 4.03459549706 #latvec3   
14   
15 ATOMIC_POSITIONS   
16 Direct   
17   
18 Al #label   
19 0 #magnetism   
20 4 #number of atoms   
210 0 omo 0 0   
22 0.5 0.5 0 m 0 0 0   
230.5 0 0.5m 0 0 0   
240 0.5 0.5 m 0 0 0

## 用Phonopy产生需要计算的超胞及相应微扰构型

这里我们使用有限位移方法计算声子谱，因此需要对晶格进行扩胞并对原子位置进行微扰。执行如下命令即可生成222的扩胞并产生微扰结构：

```batch
1 phonopy -d --dim="2 2 2" --abacus
```

这一步 phonopy 会根据晶格对称性自动产生相应个数的微扰结构。由于FCC的晶格对称性较强，因此这个例子只产生一个微扰结构：STRU-001。这里类似K 点的对称性分析，晶体结构对称性越强，所需的微扰结构就越少，对称性稍差的体系一般会产生多个微扰结构。

经验性设置：1）扩胞越大，计算结果越精确，但是计算量也会上升，一般来说扩胞三个方向的cell长度均在10-20A是比较合适的；2）对于优化后的晶胞（复杂体系)，原子位置可能不处于高对称点上，phonopy 可能计算存在一定的误差，可以使用 Matertial Studio 等软件把对称性加回去之后，再做上述步骤，这样能够得到准确的声子谱数据（保证计算出来的声子谱满足体系的对称性特征)。

## 产生FORCE_SET文件

接着用 ABACUS 计算原子受力，其中需要注意的是calculation 需要设置为 scf，并且设置cal_force为1，因为这一步目的是输出原子受力。

小技巧：即为了计算不同的微扰结构的受力，可以在 INPUT 里添加关键字 stru_file 来指定 STRU文件的路径和文件名：stru_file./STRU-001

INPUT内容如下：

INPUT_PARAMETERS   
2 #Parameters (1.General)   
3 suffix Al-fcc   
calculation scf   
esolver_type ksdft   
6 symmetry 1   
pseudo_dir ./psp   
8 orbital_dir ./psp   
9 cal_stress 1   
10 cal_force 1   
11 stru_file STRU-001   
12   
13 #Parameters (2.Iteration)   
14 ecutwfc 100   
15 scf_thr 1e-7   
16 scf_nmax 50   
17   
18 #Parameters (3.Basis)   
19 basis_type lcao   
20 gamma_only 0   
21   
22 #Parameters (4.Smearing)   
23 smearing_method mp   
24 smearing_sigma 0.015   
25   
26 #Parameters (5.Mixing)

27 mixing_type pulay   
28 mixing_beta 0.7   
29 mixing_gg0 1.5

算完之后用以下命令产生FORCE_SET文件：

1 phonopy -f ./disp-001/0ut\*/running_scf.log ./disp-002/0u\*/ruing_scf.log...

即要指定所有微扰构型算完之后的running_scf.log文件位置。如果运行有错，需要首先检查是否所有构型都已正常结束，且其中有力输出（可以找“FORCE”来确认）。

## 设置band.conf文件计算得到声子谱

执行如下命令：

```batch
1 phonopy -p band.conf --abacus
```

band.conf 内容如下：

```python
1 ATOM_NAME = Al
2 DIM = 2 2 2
3 MESH = 8 8 8
4 PRIMITIVE_AXES = 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0
5 BAND = 1 1 1 1/2 1/2 1 3/8 3/8 3/4 000 1/2 1/2 1/2
6 BAND_POINTS = 101
7 BAND_CONNECTION = .TRUE.
```

这一步结束之后会有band.yaml文件输出

以上参数在Phonopy 的线上文档中均有详细说明，这里再进行简单概述：

。 ATOM_NAME：指定结构文件中的元素种类。

·DIM：扩胞的大小，需要跟3.2用 Phonopy产生需要计算的超胞及相应微扰构型中的"dim”一致。

·MESH:q点的采样网格。'888‘意味着采用8x8x8的q点网格，默认以（0，0，0）为中心。

·PRIMITIVE_AXES：输入晶胞到目标原胞的转换矩阵，并将根据原胞基矢量作为声子计算的坐标系。这里是FCC 的原胞转换矩阵。

·BAND：采样能带的q点路径。不同晶格的高对称点不同，具体可以使用 Seek-path，自动生成q点路径。

。 BAND_POINTS：给出了包括能带路径末端的采样点的数量。

BAND_CONNECTION：在能带交叉处辅助连接能带。

## 绘制声子谱

本教程使用gnuplot 绘制声子谱，在Ubuntu上gnuplot 的安装如下：

1 sudo apt-get install gnuplot

用如下命令输出 gnuplot 格式的声子谱，并使用 gnuplot 绘制声子谱并存为 Al-FCC_plot.png:

1 phonopy-bandplot --gnuplot > pho.dat   
2 gnuplot plot_pho.gp

plot_pho.gp 内容如下:

1 set terminal pngcairo size 1920，1080 font 'Arial，36'## 格式，大小和字体   
2 set output "Al-FCC_plot.png" ### 输出的文件名   
3   
4 set ylabel 'Frequency (THz)'   
5   
6 set ytics 2   
7 unset key   
8   
9 x1 = 0.13115990   
10 x2 = 0.17753200   
11 x3 = 0.31664810   
12 xmax = 0.43023590   
13 ymin = 0   
14 ymax = 12   
15   
16 set xrange [0:xmax]   
17 set yrange [ymin:ymax]   
18 set xtics ("{/Symbol G}" 0， "X" x1, "K" x2，"{/Symbol G}" x3, "L" xmax)   
19 set arrow 1 nohead from x1,ymin to x1,ymax lt 2   
20 set arrow 2 nohead from x2,ymin to x2,ymax lt 2   
21 set arrow 3 nohead from x3,ymin to x3,ymax lt 2   
22   
23 plot 'pho.dat' using 1:(\\$2) w l lw 3

FCC Al的声子谱:

<!-- image-->

也可使用Origin 绘图，pho.dat的第一列就是上图的横轴（K点路径)，其中高对称K点位置见pho.dat的第二行，第二列就是上图的纵轴(声子频率，单位THz)。

## 6.8 ABACUS+ShengBTE 计算晶格热导率

## 6.8.1介绍

本教程旨在介绍采用ABACUS（基于ABACUS 3.2.0版本）做密度泛函理论计算，并且结合 ShengBTE软件计算晶格的热导率的流程。其中，整个计算过程中还用到了：1）采用Phonopy程序来计算二阶力常数，2）采用ASE程序进行原子结构的转换，3）采用 ShengBTE的thirdorder程序计算三阶力常数，4）最后使用 ShengBTE来计算材料的晶格热导率。

上述提到了一些需要结合的外部软件，这里推荐大家阅读这些软件的相关文档和说明：

ShengBTE:https://bitbucket.org/sousaw/shengbte/src/master/

phonopy: http://abacus.deepmodeling.com/en/latest/advanced/interface/phonopy.html

ASE:http://abacus.deepmodeling.com/en/latest/advanced/interface/ase.html

thirdorder: https://bitbucket.org/sousaw/thirdorder/src/master/

## 6.8.2准备

ABACUS 的软件包中提供了一个 ABACUS+ShengBTE 计算晶格热导率的算例，可以从Gitee上下载。算例中包含采用数值原子轨道的 LCAO（Linear Combination of Atomic Orbitals）和采用平面波基矢量的 PW（PlaneWave，平面波）两个文件夹。每个文件夹下分别又包含了2nd、3rd和shengbte这三个文件夹，分别保存了使用phonopy计算二阶力常数（2nd）、thirdorder 计算三阶力常数（3rd）和 ShengBTE计算晶格热导率（shengbte）的相关文件。

## 6.8.3流程

以LCAO 文件夹为例，我们这里提供的测试案例是包含2个原子的金刚石结构 Si结构，采用的模守恒赝势是 Si_ONCV_PBE-1.0.upf，以及原子轨道文件采用的是Si_gga_7au_100Ry_2s2pld.orb（GGA 泛函，7au截断半径，100 Ry 能量截断，以及包含2s2pld的DZP 轨道)。

## 计算二阶力常数

要计算二阶力常数，除了ABACUS之外，还需要结合Phonopy 和ASE。首先，进人2nd文件夹。结构优化

做晶格热导率计算之前要先对模拟的材料体系的进行原子构型的优化。下面是采用ABACUS 做结构优化(relax）后得到的原子构型文件 STRU。在这个例子里，为了简单起见，结构优化过程采用的是2\*2\*2的布里渊区k点采样，平面波的能量截断值ecut（LCAO里面也用到了平面波基矢量）为100Ry，注意实际计算中应该要采用更收敛的k点采样。

1 ATOMIC_SPECIES   
2 Si 28.0855 Si_ONCV_PBE-1.0.upf   
3   
4 NUMERICAL_ORBITAL   
5 Si_gga_7au_100Ry_2s2p1d.orb   
6   
7 LATTICE_CONSTANT   
8 1.88972612546   
9   
10 LATTICE_VECTORS   
11 0 2.81594778072 2.81594778072 #latvec1

12 2.81594778072 0 2.81594778072 #latvec2   
13 2.81594778072 2.81594778072 0 #latvec3   
14   
15 ATOMIC_POSITIONS   
16 Direct # direct coordinate   
17   
18 Si #label   
19 0 #magnetism   
20 2 #number of atoms   
21 0.875 0.875 0.875 m 0.00   
22 0.125 0.125 0.125 m 000

注意：第一行Si的质量28.0855在计算中不起作用。

计算二阶力常数

调用 Phonopy软件产生需要计算的超胞及相应微扰的多个原子构型，命令如下：

```batch
1 phonopy setting.conf --abacus -d
```

其中 setting.conf 文件的内容为：

1 DIM = 2 2 2   
2 ATOM_NAME = Si

这里我们采用的 Si的例子只需要产生1个微扰构型 STRU-001即可，对所有微扰构型（这里Si的例子只有1个）进行 SCF计算（SCF 代表 Self-consistent field，这里代表进行密度泛函理论的电子迭代自洽计算）获得原子受力，算完之后用以下命令产生FORCE_SET文件：

```batch
1 phonopy -f OUT.DIA-50/running_scf.log
```

小技巧：在 ABACUS 的输入文件 INPUT中可以设置变量 stru_file，该变量对应的原子构型文件为 STRU-001则 ABACUS 可以直接读取该结构文件。

下一步，设置band.conf 文件计算得到声子谱以及二阶力常数：

```batch
1 phonopy -p band.conf --abacus
```

这里出现的 band.conf 文件，其内容如下（具体参数含义可以查看 Phonopy 说明文档):

1 ATOM_NAME = Si   
2 DIM = 2 2 2   
3 MESH = 8 8 8   
4 PRIMITIVE_AXES = 1 0 0 O 1 0 0 0 1   
5 BAND = 0.0 0.00.0 0.50.0 0.5 0.625 0.25 0.625,0.375 0.375 0.75 00 0.00.0 0.50.5 0.5   
6 BAND_POINTS = 101   
7 BAND_CONNECTION = .TRUE.   
8 FORCE_CONSTANTS = WRITE   
9FULL_FORCE_CONSTANTS = .TRUE.

这一步结束之后，Phonopy 软件会产生 band.yaml（用于绘制声子谱）和 FORCE_CONSTANTS 文件。其中，FORCE_CONSTANTS 文件包含的数据即为二阶力常数，注意这里务必设置FULL_FORCE_CONSTANTS =.TRUE.，输出全部的二阶力常数，否则 ShengBTE 读取数据会报错。

此外，可以使用如下命令输出 gnuplot格式的声子谱，用于绘制声子谱：

```batch
1 phonopy-bandplot --gnuplot > pho.dat
```

后处理

注意 ShengBTE 软件要求FORCE_CONSTANTS_2ND 文件里数据的单位为eV/A^2，但是ABACUS 结合 phonopy计算的FORCE_CONSTANTS 单位为eV/(Axau)，其中au 是原子单位制，1 au=0.52918 A。可以使用 2nd 目录下提供的au2si.py 脚本进行单位转换，生成 FORCE_CONSTANTS_2ND 文件，命令如下:

1 python au2si.py

在 shengbte 文件夹中提供了FORCE_CONSTANTS_2ND 文件供参考计算结果。

## 计算三阶力常数

要计算三阶力常数，需要结合thirdorder程序，计算后输出三阶力常数文件FORCE_CONSTANTS_3RD。但是，thirdorder目前只支持读取VASP 和QE 的输入输出文件。因此，这里我们是通过将 ABACUS 的结构文件和输出受力分别转换为POSCAR和 vasprun.xml来使用thirdorder，请先进人3rd文件夹，具体步骤将在以下叙述。获得微扰构型

首先将 ABACUS 软件进行结构优化（relax）后的 STRU 文件转化为POSCAR（目录下已给出转化过的POSCAR，或者需要自己动手进行这个转换）。

之后，运行thirdorder_vasp 程序，产生微扰过后的一系列原子构型文件3RD.POSCAR.\*，例如这个例子一共产生了40个构型:

```batch
1 thirdorder_vasp.py sow 2 2 2 -2
```

运行 pos2stru.Py，将上述POSCAR 转化为 STRU文件，注意该脚本里调用了ASE 软件包的函数（需提前安装好ASE):

```batch
1 python pos2stru.py
```

注意：这里不能调用dpdata软件进行转化。因为dpdata 会强制将晶格改为下三角矩阵，相当于旋转了晶格，会导致原子间受力方向也相应旋转，从而发生错误。

## 计算微扰构型的原子受力

可以参考目录下run_stru.sh使用脚本批量产生SCF-\*文件夹并提交计算，这里需要采用ABACUS对 40个原子构型分别进行 SCF计算，会有些耗时。建议每个SCF 单独在SCF-\*文件夹内运行，这里的INPUT中的 scf_thr需要至少小到1e-8才能得到收敛的结果。

计算完成后，运行aba2vasp.py，将 ABACUS 计算的原子受力包装成 vasprun.xml格式，放置在每个 SCF-\*文件夹中，命令如下：

1 python aba2vasp.py

vasprun.xml格式示意：

最后执行如下命令：

即可得到三阶力常数文件FORCE_CONSTANTS_3RD。在shengbte文件夹中提供了FORCE_CONSTANTS_3rd文件供参考计算结果。

## 运行ShengBTE得到晶格热导率

进入shengbte 文件夹，里面已经准备好CONTROL（ShengBTE 的参数文件）、FORCE_CONSTANTS_2ND（二阶力常数文件）、FORCE_CONSTANTS_3RD（三阶力常数文件）这三个文件，使用如下命令运行 ShengBTE即可得到晶格热导率，其中Ref文件夹中给出了计算结果供参考：

1 mpirun -n 10 ShengBTE

## 6.8.4结尾

对于ABACUS中使用平面波（PW）来做 ShengBTE的计算也是采用以上类似的流程，但要注意使用平面波时，计算三阶力常数的INPUT中 scf_thr需要至少小到1e-12。通过计算结果可以发现，PW和LCAO基组计算出的 Si的晶格热导率是接近的，300K下均在100W/(mK)左右，而实验中 Si在300K的热导率在150W/(m K)附近。这是因为作为教学例子，这里使用的是2\*2\*2的扩胞以及2\*2\*2的K点，导致计算结果偏小，实际科研中需要测试扩胞的大小以及K 点的采样方案来达到收敛的结果。以上就是 ABACUS(3.2.0)+ShengBTE 计算晶格热导率的全部流程，如果有什么问题，欢迎通过邮件联系。

## 6.9 ABACUS+Wannier90使用教程

## 6.9.1教程使用说明

Wannier函数是周期性晶格势场中局域表象下的波函数，Wannier 函数在1937年由Gregory H.Wannier提出,它可以通过傅立叶变换和布洛赫波函数互相转换。Wannier函数对于理解半导体、过渡金属、分子晶体等价电子局域程度比较高的凝聚态系统有重要帮助，因而在物理中有广泛的应用。

在 ABACUS 等密度泛函理论（DensityFunctionalTheory，简称DFT）软件中，单电子近似下的布洛赫波函数可以通过自洽迭代求解 Kohn-Sham 方程得到。但是布洛赫波函数存在规范不确定性，进而导致 Wannier 函数[1-4]也不是唯一的。1997年到 2001年间，Nicola Marzari、David Vanderbilt和Ivo Souza等人发展了最大局域化的Wannier函数（Maximally localized Wannier function,简称 MLWF）的方法[1,2,4]，提出可以通过优化Wannier表象下的电子位置的方差（即 spread函数）来确定一个规范，在该规范下Wannier函数是最局域的；该方法可以在 DFT电子结构计算完之后以后处理的方式实现计算MLWF。2008年，用于产生MLWF的Wannier90 软件发布，并在随后几年获得了广泛的应用。

本教程分为两部分，第一部分最大局域化Wannier函数方法简介介绍最大局域化Wannier 函数的算法，主要介绍1997年和 2001年的两篇 PRB 文章中提出的优化算法，侧重物理与算法原理；第二部分即本文档，是Wannier90 软件的ABACUS 接口说明，主要介绍如何使用ABACUS 计算Wannier90需要的输入文件，侧重具体软件使用。

## 6.9.2Wannier90 软件介绍

Wannier90软件[5-7]是 Arash A.Mostofi 同Marzari、Vanderbilt等人在 2008 年写的做Wannier函数最局域化的Fortran语言写成的软件，目前是做Wannier函数最局域化的主流软件。在Wannier函数方面，Wannier90 软件可以输出Wannier 函数的中心（负电中心），也可以画出Wannier函数的等值面图像。除了做Wannier 函数相关的计算，Wannier90 还具有计算能带与态密度、Berry Phase、输运性质等方面的功能，程序的下载和使用方法可以参考其官网：Wannier90。

## 6.9.3使用ABACUS准备Wannier90需要的文件

目前的 ABACUS 平面波基组功能支持Wannier90 接口，LCAO 的接口还在完善中。

使用Wannier90 接口不需要编译ABACUS 过程中做额外操作。

## ABACUS软件的输人参数解释

<table><tr><td rowspan=1 colspan=1>参数名</td><td rowspan=1 colspan=1>默认值</td><td rowspan=1 colspan=1>变量类型</td><td rowspan=1 colspan=1>参数解释</td></tr><tr><td rowspan=1 colspan=1>towannier90</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>是否打开Wannier90接口的计算功能</td></tr><tr><td rowspan=1 colspan=1>nnkpfile</td><td rowspan=1 colspan=1>seedname.nnkp</td><td rowspan=1 colspan=1> string</td><td rowspan=1 colspan=1>Wannier90产生的nnkp文件的路径,nnkp文件包含了k 点邻居关系等计算mmn 和amn 文件必要的信息。</td></tr><tr><td rowspan=1 colspan=1>wannier_spin</td><td rowspan=1 colspan=1>up</td><td rowspan=1 colspan=1>string</td><td rowspan=1 colspan=1>输出向上/向下的波函数及其内积</td></tr><tr><td rowspan=1 colspan=1>out_wannier_mmn</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>是否输出mmn文件，mmn文件存储了Mmnmathbf(k,b))，即布洛赫波函数之间的内积。mmn 文件在计算Wannier最后域化时必须输出。</td></tr><tr><td rowspan=1 colspan=1>out_wannier_amm</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>是否输出amn文件，amn文件存储了Amn，即初猜的Wannier函数和布洛赫波函数之间的内积。在Wannier最后域化计算中最好有该文件（可以加快优化迭代的收敛)，也可以没有。</td></tr><tr><td rowspan=1 colspan=1>out_wannier_eig</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>是否输出eig文件，eig文件存储了各个k点、各个能带的本征能量。在计算输出能带时需要输出，如果不需要输出能带则不需要输出该文件。</td></tr><tr><td rowspan=1 colspan=1>out_wannier_unk</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>是否输出UNK文件，UNK文件存储了布洛赫波函数unlmathbf(k)。计算Wannier函数最后域化，输出Wannier 函数的中心和 spread不需要波函数信息。只有画Wannier函数图像时需要UNK文件。</td></tr><tr><td rowspan=1 colspan=1>out_wannier_wvfn_formatted</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>输出的UNK文件是否是文本文件。非文本文件即二进制文件，输出速度会比文本文件快很多。</td></tr></table>

在运行ABACUS+wannier90程序前，需要准备自洽计算（SCF）的输入文件、非自洽计算（NSCF）的输入文件和Wannier90 的输入文件，ABACUS 的输入文件有如下注意事项：

1.自洽计算：需要注意INPUT中一定要指定out_chg 参数为1，即输出电荷密度，以供给非自洽计算计算使用；

2.非自洽计算：需要注意INPUT 中需要把init_charge 参数设置为file，即读取文件中的电荷密度；一定要指定towannier90 参数为1，同时指定nnkpfile 参数为Wannier90产生的 nnkp 文件的地址；另外需要注意 KPT文件中的布里渊区k点要和Wannier90输入文件中的k点一模一样，所以最好手动输入K点坐标指定;

3．非自洽计算：INPUT 中有 out_wannier_mmn、out_wannier_amn、out_wanier_eig、out_wannier_unk、out_wannier_wvfn_formatted五个控制 Wannier90 相关文件输出的参数，默认值都是1，其中out_wannier_eig 和out_wannier_mmn 是必须打开的，out_wannier_unk 和out_wannier_amn 在需要使用 Wannier90绘制波函数图时需要打开，out_wannier_wvfn_formatted控制输出的UNK文件是文本文件还是二进制文件，输出文本文件的时间成本和存储空间都会比较大，输出二进制文件的时间开销和存储开销则会相对较小。

## 运行流程

这里给出了运行 ABACUS+wannier90的一个脚本文件，结合 examples/interface_wannier90/中的例子讲解结合ABACUS 和Wannier90计算最局域化Wannier函数的完整运行流程，这里的流程对平面波基组和原子轨道基组均适用，原子轨道基组的NSCF会比平面波基组慢一些。

export W90=wannier90.x  
2 export ABACUS=abacus  
3 export NP=12  
4  
5 # 1．运行 Wannier90 软件，生成 nnkp 文件；  
6 cd wannier

```shell
7 mpirun -np $NP $W90 -pp diamond.win
8
9#2．运行 ABACUS 软件，执行电子结构的自洽计算;
10 cd ../ABACUS
11 cp INPUT-scf INPUT
12 cp KPT-scf KPT
13 mpirun -np $NP $ABACUS >> scf.out
14
15 #3．将 nnkp 文件放在 ABACUS 的非自洽计算的 INPUT 文件中指定的位置，运行 ABACUS 软件，执行非自洽计算；
16 cp ../wannier/diamond.nnkp .
17 cp INPUT-nscf INPUT
18 cp KPT-nscf KPT
19 mpirun -np $NP $ABACUS >> nscf.out
20
21#4．将非自洽计算生成的 Wannier90 所需要的文件（mmn、amn、eig、UNK）放在 Wannier90 的运行路径下，运行 Wannier90 软
22 cd ../wannier
23 cP/ABACUS/OUT.ABACUS/diamond.amn../ABACUS/OUT.ABACUS/diamond.mn../ABACUS/OUT.ABACUS/diamond.eig../ABACUS
24 mpirun -np $NP $W9O diamond.win
```

在这个例子中，Wannier90 的输人文件 diamond.win 中指定要求画出Wannier 函数的图像，运行结束之后会得到diamond_0000[1-4].xsf 四个文件，存有Wannier 函数的实空间格点数值，放在 VESTA 中可以画出如下图的Wannier函数的等值面。

<!-- image-->

## 运行时间参考

由于Wannier90 接口只涉及上述步骤中的NSCF 计算，所以这里只提供NSCF部分的时间统计。

在12个CPU（型号：Intel(R) Xeon(R)Bronze 3104 CPU @ 1.70GHz）上并行运行样例，UNK文件采用二进制格式输出，平面波基组耗时4秒，原子轨道基组耗时98 秒。

## 样例文件

下载地址为（国内gitee):

https://gitee.com/mcresearch/abacus-user-guide/tree/master/examples/interface_Wannier90

## 6.9.4参考文献

Nicola Marzari, Arash A.Mostofi, Jonathan R.Yates,Ivo Souza,and David Vanderbilt, Maximaly localized Wannier functions: Theory and applications,Rev.Mod.Phys.84,1419 (2012)．（关于最大局域化Wannier 函数的综述文献）

Nicola Marzari and David Vanderbilt, Maximaly localized generalized Wannier functions for composite energy bands,Phys.Rev.B 56,12847(1997)．（提出最大局域化Wannier 函数方法的文献）

Blount,E.I.,Solid State Phys．13,305(1962)．（关于 Bloch 状态的诸多基础性质的讨论，最大局域化中(nR|r|m0)在布洛赫表象下的表达式来源于此)

Ivo Souza,Nicola Marzari,and David Vanderbilt, Maximally localized generalized Wannier functions for entangled energy bands,Phys.Rev.B65,035109(2001)．（从纠缠能带中解耦出部分能带，做最大局域化 Wannier 函数的文 献)

Arash A.Mostofi,JonathanR.Yates,Young-SuLee,Ivo Souza,David Vanderbilt,Nicola Marzari, wannier9O: A tool for obtaining maximall-localised Wannier functions,Comput.Phys. Commun.,178,9,685 (2008).(发布 Wannier90软 件的文献)

Giovanni Pizzi etal. Wannier90 as acommunity code: new features and applications. J.Phys.: Condens.Matter,32, 165902 (2020).(Wannier90 软件近期的更新)

ABACUS 的 Wannier90 接口的英文文档：Wannier90 ABACUS documentation

## 6.10 ABACUS+Candela 使用教程

## 6.10.1介绍

本教程旨在介绍采用Candela分析ABACUS（基于3.2.4版本）分子动力学轨迹，计算径向分布函数（RadialDistribution Function，简称RDF），静态结构因子（Static Structure Factor，简称SSF），离子-离子动态结构因子(Ion-ion Dynamic Structure Factor，简称 DSF）以及均方差位移（Mean Square Displacement，简称MSD）流程。

上述性质的具体描述可以参考：Qianrui Liu et al 2020 J.Phys.：Condens.Matter 32 144002，径向分布函数(RDF）见文中公式（12)，静态结构因子（SSF）见公式（13），离子-离子动态结构因子（DSF）见公式（15，16,17）以及均方差位移（MSD）见公式（18)。

Candela 全称Collction of ANalysis DEsigned for Large-scale Atomic simulations，目前支持分析 QE、ABACUS、LAMMPS 和 VASP 的分子动力学轨迹，Github 主页：https://github.com/MCresearch/Candela

## 6.10.2准备

## 下载并安装Candela

1 git clone https://github.com/MCresearch/Candela.git  
2 cd Candela  
3 make -j4 # 使用 intel oneapi 编译器

若使用IntelOneapi编译器，按照上述流程操作，在编译完成后即可在bin目录下看到candela可执行文件若使用其他编译器，需要按照Candela目录下Makefile.vars 修改相应的CXX

## 下载例子

可以从 Gitee上下载。在网页右侧点击克隆/下载->下载 ZIP得到算例，或者在linux 终端执行如下命令得到算例：

1 git clone https://gitee.com/mcresearch/abacus-user-guide.git

下载完成后解压，之后进入abacus-user-guide/examples/candela 文件夹

算例中包含 RDF、SSF、DSF、MSD 以及MSD_ntype_2五个文件夹和 MD_dump 文件。其中MD_dump 是32原子的铝的MD 轨迹

## 6.10.3流程

## 计算径向分布函数

进入RDF文件夹，INPUT文件即为Candela 的输入文件，具体如下：

```asm
1 calculation pdf
2 geo_in_type ABACUS
3 geo_directory../MD_dump
4 geo_1 0
5 geo_2 100
6 geo_interval 2
geo_ignore 50
8 geo_out Al_rdf .txt
9
10 ntype 1
11 natom 32
12 rcut 2.3
13 dr 0.01
```

以上参数在Candela 的线上文档中均有详细说明，这里再进行简单概述：

。 calculation:设置为pdf(PairDistribution Function)即计算径向分布函数

·geo_in_type:读取的 MD 轨迹的格式，目前支持分析 pw.x(QE2)，cp.x (QE)，ABACUS (ABACUS)、LAMMPS(LAMMPS)和 VASP (VASP)

geo_directory:MD 轨迹的位置

geo_1:MD轨迹起始的索引

geo_2:MD轨迹结束的索引

。 geo_interval:Candela 读取MD 轨迹的间隔

。 geo_ignore:需要跳过的MD轨迹帧数

geo_out:输出的文件名，默认为result.dat

。 ntype:原子种类

natom:原子数

rcut:计算RDF 的截断半径，一般取晶格的一半，单位为 Angstrom

dr:计算RDF的r的间隔，单位为Angstrom

执行如下命令：

1 mpirun -n 2 candela

即可得到RDF，输出到Al_rdf.txt，其中第一列为r，单位为 Angstrom，第二列即为对应的RDF，第三列为对第二列积分的结果。

## 静态结构因子

进入SSF文件夹，INPUT文件具体如下：

```csv
1 calculation ssf
2 geo_in_type ABACUS
3 geo_directory ../MD_dump
4 geo_1 0
5 geo_2 100
6 geo_interval 2
geo_ignore 50
8
9 ssf_out Al_ssf.txt
10 ntype 1
11 natom 32
12 struf_dgx 1.32656
13 struf_dgy 1.32656
14 struf_dgz 1.32656
15 struf_ng 6
```

以上参数在Candela 的线上文档中均有详细说明，这里再进行简单概述：

。 calculation:设置为ssf，即计算静态结构因子

ssf_out:输出的文件名

：struf_dgx:倒空间中的间隔，一般取2π/a，其中α为x方向的晶格长度，单位为Angstrom-1

struf_dgy:一般取2π/b，其中b为y方向的晶格长度，单位为Angstrom-1

。 struf_dgz:一般取2π/c，其中c为z方向的晶格长度，单位为Angstrom-1

struf_ng:上述倒空间中的间隔的数量

执行如下命令：

1 mpirun -n 2 candela

即可得到 SSF，输出到Al_ssf.txt（原始的计算结果）以及 sm-Al_ssf.txt（平滑的计算结果)，其中第一列为q，单位为Angstrom-1，第二列即为对应的 SSF。

## 离子-离子动态结构因子

进入DSF 文件夹，离子-离子动态结构因子需要对中间散射函数（Intermediate Scattering Function）进行傅里叶变换得到，因此需要先计算中间散射函数，INPUT文件具体如下：

1 calculation isf2   
2 geo_in_type ABACUS   
3 geo_directory ../MD_dump

4 geo_1 50   
5 geo_2 100   
6 geo_interval 1   
7   
8 isf_outfile isf.txt   
9 ntype 1   
10 natom 32   
11 isf_nt1 11   
12 isf_nt2 40   
13 dt_snapshots 0.00006   
14   
15 isf_target_q 2.65   
16 isf_dgx 1.32656   
17 isf_dgy 1.32656   
18 isf_dgz 1.32656   
19 isf_ngx 6   
20 isf_ngy 6   
21 isf_ngz 6

以上参数在Candela的线上文档中均有详细说明，这里再进行简单概述：

。 calculation:设置为isf2，即计算中间散射函数

。 isf_outfile:输出的文件名

。 isf_nt1:中间散射函数的时间长度

·isf_nt2:用于平均中间散射函数的时间长度，注意需要控制 interval\*(nt1+nt2)<=geo_2-geo_1+1

。 dt_snapshots:MD 轨迹每一帧之间的时间步长，单位为 ps

isf_target_q:中间散射函数计算的目标q，单位为Angstrom-1

·isf_dgx:倒空间中的间隔，一般取2π/a，其中a为x方向的晶格长度，单位为Angstrom-1

isf_dgy:一般取2π/b，其中b为y方向的晶格长度，单位为Angstrom-1

isf_dgz:一般取2π/c，其中c为z方向的晶格长度，单位为Angstrom-1

。 isf_ngx:上述倒空间中x方向的间隔的数量

。 isf_ngy:上述倒空间中y方向的间隔的数量

1 mpirun -n 2 candela

即可得到ISF，输出到isf.txt，其中第一列为时间，单位为ps，第二列即为对应的 ISF。

接着执行如下命令得到离子-离子动态结构因子，其中PathtoCandela指的是Candela的下载目录：

```batch
1 python PathtoCandela/examples/e3_dsf/onedsf.py 0.00006 0.0006
```

其中第一个参数0.00006=dt_snapshots \* geo_interval，为ISF的时间步长，单位为ps；第二个参数0.0006,为对 ISF做傅里叶变换的总时长，单位为ps。这里onedsf.py默认读取 isf.txt，并输出到dsf.txt，其中第一列为W，单位为meV，第二列即为对应的DSF。

## 均方差位移

单种原子

进入MSD文件夹，INPUT文件具体如下：

```csv
1 calculation msd_multiple
2 geo_in_type ABACUS
3 geo_directory ../MD_dump
4 geo_1 0
5 geo_2 100
6 geo_interval 1
geo_ignore 50
8
9 ntype 1
10 natom 32
11 msd_n 2
12 msd_t0 0.003
13 msd_t 0.0015
14 msd_dt0 0.0015
15 msd_dt 0.00006
16 msd_natom 32
```

以上参数在Candela的线上文档中均有详细说明，这里再进行简单概述：

。 calculation:设置为msd_multiple，即计算均方差位移

msd_n:计算均方差位移的段数，目的是将MD轨迹分成若干段，分别计算均方差位移

。 msd_t0:计算均方差位移的起始时间，单位为ps

. msd_t:每段均方差位移的时长，单位为ps

·msd_dt0:两段均方差位移起始位置的间隔，单位为ps。若设置等于msd_t，则每段均方差位移之间的时间距离为0

·msd_dt:获取的MD轨迹每一帧之间的时间步长（等于 geo_interval \*dt，其中dt 为MD轨迹每一帧之间的时间步长)，单位为 ps

msd_natom:需要设置为计算的原子数

执行如下命令：

1 mpirun -n 2 candela

即可得到MSD，输出到 MSD_each.txt和 MSD_total.txt。其中 MSD_each.txt 第一列为时间，单位为 ps，第二列到最后一列即为对应的每段的MSD，单位为 Angstrom2；MSD_total.txt第一列为时间，单位为ps，第二列为平均的每段的MSD，单位为Angstrom2

多种原子

进入MSD_ntype_2文件夹，INPUT文件具体如下：

1 calculation  msd_multiple   
2 geo_in_type ABACUS   
3 geo_directory MD_dump   
4 geo_1 0   
5 geo_2 100   
6 geo_interval 1

7 geo_ignore 50   
8   
9 ntype 2   
10 natom 17   
11 natom1 16   
12 natom2 1   
13 id1 C   
14 id2 Li   
15   
16 msd_n 2   
17 msd_t0 0.003   
18 msd_t 0.0015   
19 msd_dt0 0.0015   
20 msd_dt 0.00006   
21 ele_select   
22 msd_natom 16  
以上参数在Candela的线上文档中均有详细说明，与单种原子不同的参数如下：  
。 natoml、natom2：不同种类原子的个数  
·id1、id2：不同种类原子的名称，需要与读入的MD_dump 中原子名称一致  
ele_select：计算MSD的原子名称，需要与id1或id2对应

多种原子的计算过程与结果与单种原子并无区别。

## 6.11 ABACUS+USPEX 接口教程

## 6.11.1 背景介绍

近年来，随着高性能计算机的快速发展，晶体结构预测算法已然成为计算材料学的核心研究手段之一，极大地促进了新材料设计与研发的进程。Artem R.Oganov教授课题组开发的进化类晶体结构预测算法USPEX 是当前主流的全局搜索类算法之一。软件自 2004年发布以来，该算法已被广泛应用于科研人员的研究工作中。晶体结构预测算法的主要作用是生成结构以及控制结构的演进，结构预测的效率及可靠性除却算法本身的影响外还对结构弛豫软件有极强的依赖性。ABACUS（原子算筹）作为一款国产开源密度泛函理论软件，同时兼具高精度和高效率，可适用于从小体系到上千原子体系的电子结构优化、原子结构弛豫、分子动力学等计算。因此实现晶体结构预测算法 USPEX（htps://spex-team.org/en）和第一性原理软件 ABACUS 的结合会给广大研究人员在结构预测研究中带来便利。

本教程旨在为大家提供一个USPEX-ABACUS 接口的实现教程，方便大家程序配置。本教程以单质硅（Si)的结构预测为例,介绍了该接口使用所需的输入文件和参数设置。

如需了解更多关于USPEX的使用信息，请参考USPEX官方文档：USPEXDocumentation USPEX

## 6.11.2接口介绍

当前发行的USPEX提供了分别以matlab和python为核心编程语言的两个版本，前者的版本编号为USPEX-vXX（如USPEX_v10.5），后者的版本编号为USPEX-YEAR-XX（如USPEX 2023.0.2），需要注意的是当前只有python 版本可以自定义外部接口。该教程中的接口是在USPEX2023.0.2和 ABACUS 3.3.0版本进行编写测试的，如发生版本变更引起的异常，根据报错再行修改。

首先通过USPEX官网（USPEX下载 USPEX）下载USPEX 2023.0.2软件（注册后即可免费下载)，解压后根据所使用的python 版本,选择软件压缩包进行安装,例如 pip installuspex-2023.0.2-cp310-cp310-linux_x86_64.whl。注：若系统提示需要安装虚拟环境，可以在USPEX2023.0.2目录下激活虚拟环境后重新进行安装。

1 python -m venv tutorial-env   
2 source tutorial-env/bin/activate

该接口共包含三个文件

(1)USPEX/components.py（对接口进行注册）

(2) USPEX/Stages/Interfaces/ASEInterfacesAdapter.py（完成 ASE 接口中的读 read 和写 write 功能）

(3)USPEX/Stages/Interfaces/ABACUS_Interface.py（接口实体，主要功能是准备 Abacus 的输人文件和读取Abacus 的计算结果）

注：以上路径中USPEX 为主目录。

上述文件可在 Github 仓库下载：htps://github.com/gxq1219/Interface_USPEX-ABACUS/tree/master

安装完成后将以上文件复制到对应的路径下，并且需要手动添加环境变量，例如:exportPATH=python3.10/site-packages/USPEX:\$PATH，然后将 USPEX 2023.0.2 目录下的 random_cell 复制到设置的环境变量路径下。该接口需要安装 ase-abacus 版本的ase库，测试版本为ase-3.23.0b1

安装 ase-abacus 接口

1 git clone https://gitlab.com/1041176461/ase-abacus.git   
2 cd ase-abacus   
3 python3 setup.py install

此次测试是在QSUB任务管理系统下进行测试,如果更换任务管理系统,应修改USPEX输入文件input.uspex中的taskManager和TM，此外需修改USPEX/Stages/TaskManager中所选任务管理方式中对任务队列状态识别的方式，如在QSUB.py 文件中应根据所使用平台修改def_parseJobID 中的内容识别出提交任务的ID，以及下文的def isReady和def isExist等函数。

QSUB.py示例（第 56行的函数，根据自己超算平台修改，功能是执行qsub sub.sh后从屏幕输出的文本中提取出 jobID):

```python
1
2 USPEX .Stages .TaskManagers .QSUB
3 ===
4
5 import logging
6 from os.path import join as pj
7 logger = logging.getLogger(__name__)
8 class QSUB:
9 shortname = 'QSUB'
10 _RUNSCRIPT = 'jobscript'
11 def __init__(self,header : str，connector):
12
13 :param header: description of params of TaskManager
14 :param connector: for remote submission
15
16 self.connector = connector
17 self.header = header
```

```python
18 def _prepareSubmission(self, C0MMAND_EXEC : str， JOB_NAME : str,
19 inputFile : str，outputFile : str,errorFile : str) -> str:
20 "
21 Preparing jobscript for submission
22 :param commandExec:
23 :param jobName:
24 :return: jobscript as string
25
26 content = -
27 for line in self.header.split('\n'):
28 if ‘ -N‘ in line:
29 logger.info('Job name found in HEADER will be overwritten')
30 elif ‘ -o ‘ in line:
31 logger.info('Output file name found in HEADER will be overwritten')
32 elif ' -e ' in line:
33 logger.info('Error file name found in HEADER will be overwritten')
34 else:
35 content += line + '\n'
36 content += f'#PBS -N {JOB_NAME}\n'
37 content += f'#PBS -o {outputFile}\n'
38 content += f'#PBS -e {errorFile}\n'
39 content += f'cd \$PBS_O_WORKDIR\n'
40 content += f'\n{COMMAND_EXEC}\n'
41 return ''.join(content)
42 async def submit(self， command: str, jobname: str, input: str, output: str, error: str，
43 calcFolder : str) -> int:
44 content = self._prepareSubmission(command, jobname， input， output， error)
45 with open(pj(calcFolder， self._RUNSCRIPT)，'wt'） as f:
46 f.write(content)
47 await self.connector.sync_l2r(pj(calcFolder， self._RUNSCRIPT))
48 returncode, out， err = await self.connector.execute(f'qsub {self._RUNSCRIPT}',
49 cwd=calcFolder)
50 logger.debug(f'process returned code {returncode}')
51 if returncode != 0:
52 logger.error(err)
53 logger.error(out)
54 jobID = self._parseJobID(out，err)
55 logger.info(f"Job ID is {jobID}")
56 return jobID
57
58 def _parseJobID(self,output : str,error : str) -> int:
59
60 :param output:
61 :param error:
62 :return: jobID
63
64 if '.mgmt' in output:
65 tmp = output.index('.mgmt')
```

---

## 第 281–300 页

66 return int(output [:tmp])   
67 elif 'job' in output:   
68 tmp = output.index('job') + 4   
69 return int(output [tmp:])   
70 elif 'comput100' in output:   
71 tmp = output.index('comput100')-1   
72 return int(output[:tmp])   
73 else:   
74 return int(output)   
75 async def isReady(self， jobID):   
76 returncode，out，err = await self.connector.execute(f'qstat {jobID}')   
77 return not (' R' in out or ' Q‘ in out)   
78 async def isExist(self，jobID):   
79 returncode，out，err = await self.connector.execute(f'qstat {jobID}')   
80 return len(out)> 0   
81 async def kill(self， jobID):   
82 returncode, out，err = await self.connector.execute(f'qdel {jobID}')   
83 #logger.info(f'Process with jobID={jobID} killed.')

## 6.11.3流程 (以 Si为例)

## 准备 USPEX的输人文件 input.uspex

```yaml
1 {
2 optimizer: {
3 type: GlobalOptimizer
4 target: {
5 type: Atomistic
6 conditions: {externalPressure: 0.00001}
7 compositionSpace: {symbols: [Si]
8 blocks:[[4]]}
9 }
10 optType: enthalpy
11 selection: {
12 type: USPEXClassic
13 popSize: 4
14 initialPopSize: 4
15 bestFrac: 0.6
16 optType: (aging enthalpy)
17 fractions: {
18 heredity: (0.4 0.7 0.5)
19 softmodemutation: (0.2 0.5 0.3)
20 randSym: (0.05 0.5 0.1)
21 randTop: (0.05 0.5 0.1)
22 }
23 }
24 }
25 stages: [abacus1 abacus2 abacus3 abacus4 abacus5]
```

26 numParallelCalcs: 4  
27 numGenerations: 25  
28 stopCrit: 10  
29 }  
30  
31 #define abacus1  
32 {type : abacus， commandExecutable : 'OMP_NUM_THREADS=1 mpirun -machinefile \\$PBS_NODEFILE -env  
33 I_MPI_HYDRA_DEBUG 3 -genv I_MPI_FABRICS shm:ofi abacus'， kresol: 0.20, taskManager: TM]  
34  
35 #define abacus2  
36 {type : abacus， commandExecutable : 'OMP_NUM_THREADS=1 mpirun -machinefile \\$PBS_NODEFILE -env  
37 I_MPI_HYDRA_DEBUG 3 -genv I_MPI_FABRICS shm:ofi abacus'， kresol: 0.16, taskManager: TM}  
38  
39 #define abacus3  
40 {type : abacus, commandExecutable : 'OMP_NUM_THREADS=1 mpirun -machinefile \\$PBS_NODEFILE -env  
41 I_MPI_HYDRA_DEBUG 3 -genv I_MPI_FABRICS shm:ofi abacus'， kresol: 0.12， taskManager: TM}  
42  
43 #define abacus4  
44 {type : abacus， commandExecutable : 'OMP_NUM_THREADS=1 mpirun -machinefile \\$PBS_NODEFILE -env  
45 I_MPI_HYDRA_DEBUG 3 -genv I_MPI_FABRICS shm:ofi abacus'， kresol: 0.08, taskManager: TM}  
46  
47 #define abacus5  
48 {type : abacus， commandExecutable : 'OMP_NUM_THREADS=1 mpirun -machinefile \\$PBS_NODEFILE -env  
49 I_MPI_HYDRA_DEBUG 3 -genv I_MPI_FABRICS shm:ofi abacus'， kresol: 0.05, taskManager: TM}  
50  
51 #define TM  
52{  
53 type : QSUB,  
54 header:"#PBS -S /bin/bash  
55 #PBS -N single  
56 #PBS -l nodes=1:ppn=16  
57 #PBS -j oe  
58 #PBS -V"  
59}

注意：ABACUS在任务提交时，应注意指定OMP_NUM_THREADS，防止内存不足引起计算错误

## 准备 Specific文件夹

所需文件：INPUT_X（使用 ABACUS 弛豫）、ATOMIC_SPECIES、NUMERICAL_ORBITAL 以及轨道和赝势文件。

INPUT 文件内容和个数由用户自行设置（需要与 input.uspex 中 stages 对应)，具体可参考uspex 官方文档。

需要注意的是：INPUT的 suffix需指定为USPEX;

需给出 ATOMIC_SPECIES 和 NUMERICAL_ORBITAL 两个文件指定所使用的 upf 文件和orb 文件。

根据计算实际需求添加 Seeds 结构(可选，非必须)

## 提交运算

```shell
1 nohup ./uspex-sub.sh >> log &
2
3 ### content of uspex-sub.sh
4
5 #!/bin/sh
6 vhile true;do
7 data >> 10g
8 uspex -r >> l1og
9 sleep 30o  ### Users can adjust this value to manipulate the frequency of the call to uspex.
10 done
```

## 计算结果

如果程序可以正常提交任务，log文件中会打印如下内容：

```csv
1 2023-08-09 16:52:47,115 - USPEX.0ptimizers.Target - INFO - Following utilities was not initialized:
2 ['PowderSpectrumAnalyzer'， 'SingleCrystalSpectrumAnalyzer'].
3 2023-08-09 16:52:47,115 - USPEX.0ptimizers.Target - INF0 - Permutation does not work when number
4 of symbols in calculation is 1.
5 2023-08-09 16:52:47,123 - USPEX.Stages.GenerationController - INF0 - Calculation initialized from
6 input parameters.
7 2023-08-09 16:52:47,441 - USPEX.Selection.USPEXClassic - INF0 - System O successfully created by
8 RandSym operator.
9 2023-08-09 16:52:47,866 - USPEX.Selection.USPEXClassic - INF0 - System 1 successfully created by
10 RandSym operator.
11 2023-08-09 16:52:48,638 - USPEX.Selection.USPEXClassic - INFO - System 2 successfully created by
12 RandSym operator.
13 2023-08-09 16:52:48,767 - USPEX.Selection.USPEXClassic - INFO - System 3 successfully created by
14 RandSym operator.
15 2023-08-09 16:52:48,788 - USPEX.Stages.Executor - INFO - System O with tag 1 will be submitted now.
16 2023-08-09 16:52:48,788 - USPEX.Stages.TaskManagers.QSUB - INFO - Job name found in HEADER will be
17 overwritten
18 2023-08-09 16:52:48,792 - USPEX.Stages.Executor - INFO - System 1 vith tag 1 will be submitted now.
19 2023-08-09 16:52:48,792 - USPEX.Stages.TaskManagers.QSUB - INFO - Job name found in HEADER will be
20 overwritten
21 2023-08-09 16:52:48,796 - USPEX.Stages.Executor - INF0 - System 2 with tag 1 will be submitted now.
22 2023-08-09 16:52:48,797 - USPEX.Stages.TaskManagers.QSUB - INFO - Job name found in HEADER vill be
23 overwritten
24 2023-08-09 16:52:48,801 - USPEX.Stages.Executor - INFO - System 3 with tag 1 will be submitted now.
25 2023-08-09 16:52:48,801 - USPEX.Stages.TaskManagers.QSUB - INF0 - Job name found in HEADER will be
26 overwritten
27 2023-08-09 16:52:48,812 - USPEX.Stages.TaskManagers.QSUB - INFO - Job ID is 37768
28 2023-08-09 16:52:48,818 - USPEX.Stages.TaskManagers.QSUB - INFO - Job ID is 37769
29 2023-08-09 16:52:48,822 - USPEX.Stages.TaskManagers.QSUB - INFO - Job ID is 37770
30 2023-08-09 16:52:48,825 - USPEX.Stages.TaskManagers.QSUB - INFO - Job ID is 37771
```

如果程序可以正常输出结果，在results1/Individuals 中会打印相关的信息，示例如下：

$$
\begin{array} { r l } { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { - \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { - \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { - \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { - \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } & { - \frac { \omega _ { 1 } } { 2 } , } \\ { \frac { 1 } { 2 } ( 2 \pi { \frac { \omega _ { 1 } } { \omega _ { 2 } } } + \frac { \omega _ { 1 } } { \omega _ { 2 } } ) } & { = } &  \frac { \omega _ { 1 } } { 2 } \end{array}
$$

## 6.11.4结语

建议大家在使用前，首先选择小体系进行测试；此外大家也可根据自己的实际需要对接口文件进行修改扩充，有相关问题可通过邮件方式沟通。

## 6.12 ABACUS+Hefei NAMD 使用教程

## 6.12.1 介绍

激发态动力学一直是凝聚态物理中的重要问题，在激发状态下，凝聚态体系中的准粒子会发生复杂的相互作用，涉及到不同时间尺度的超快过程，因此，研究凝聚态体系的激发态动力学不仅需要从时间、空间、能量和动量等多个维度对其进行描述，还需要描述不同准粒子之间的动态耦合，而目前商用的第一性原理软件无法满足这样的需求。针对这一问题，中国科学技术大学赵瑾教授团队发展了自主可控的激发态动力学软件 Hefei-NAMD（https://hefei-namd.org/code/），基于面跳跃算法（Surface Hopping）与经典路径近似，初步实现激发态动力学在时间、空间、动量、能量、自旋等多个维度上的描述，并可以研究激发态电子、空穴、激子、声子、极化子等准粒子的动态相互作用。Hefei-NAMD 程序采用模块化架构，包含单粒子动力学、自旋动力学、动量空间动力学以及GW+rtBSE模块。目前程序使用周期性边界条件，可处理上百个原子，且与密度泛函软件 VASP、Quantum Espresso 等存在接口，Hefei-NAMD 和 ABACUS v3.2 建立的接口（主要针对局域轨道算法，涉及波函数与局域轨道下的交叠矩阵等信息，用来计算非绝热耦合项和电子跃迁概率）和提供的相关案例，可以进一步拓宽Hefei-NAMD 的使用范围。

本教程旨在介绍采用ABACUS（基于ABACUS 3.2.0版本）做分子动力学计算，将结果作为Hefei NAMD软件的输入文件进行后续非绝热分子动力学计算的流程。

这里推荐大家阅读HefeiNAMD软件的相关文档和说明：

http://staff.ustc.edu.cn/\~zqj/posts/Hefei-NAMD-Training/

https://github.com/QijingZheng/Hefei-NAMD

https://github.com/vtzf/abacus-namd （这个是拥有与ABACUS 接口的 HefeiNAMD仓库，功能与上面的仓库是一样的)，第一步请先下载这个链接上面的 python 代码。

## 6.12.2计算流程

我们这里提供的测试案例是包含2个原子的金刚石结构 Si结构，采用的模守恒赝势是 Si_ONCV_PBE-1.0.upf，以及原子轨道文件采用的是 Si_gga_8au_60Ry_2s2pld.orb（GGA 泛函，8au 截断半径，60Ry 能量截断，以及包含2s2pld的DZP轨道)。

## 用ABACUS 进行分子动力学(MD)计算

输入文件 INPUT

大部分参数为做分子动力学计算所需要参数，具体含义可以参考ABACUS的分子动力学教程。为了后续用Hefei NAMD 进行非绝热分子动力学计算，ABACUS 会输出每一步分子动力学模拟的哈密顿量矩阵H、交叠矩阵S、波函数文件。因此需将相关的ABACUS 的INPUT文件中的输入参数out_wfc_lcao 和out_mat_hs都设置为1，表示输出这些物理量。计算完成后，在OUT.\*\*文件夹中，每一个分子步都对应一个文件夹MD_n，在每个文件夹中会有 data-0-H、data-0-S、LOWF_GAMMA_S1.dat 三个文件储存我们需要输出的物理量。

1 INPUT_PARAMETERS   
2 #Parameters (General)   
3 suffix autotest   
4 pseudo_dir ../../tests/PP_ORB   
5 orbital_dir ../../tests/PP_ORB   
6 nbands 8   
7 calculation md   
8   
9 #Parameters (Accuracy)

## 结构文件 STRU

1 ATOMIC_SPECIES   
2 Si 1 Si_ONCV_PBE-1.0.upf   
3   
4 NUMERICAL_ORBITAL   
5 Si_gga_8au_60Ry_2s2p1d.orb   
6   
LATTICE_CONSTANT   
8 10.2   
9   
10 LATTICE_VECTORS   
11 0.5 0.5 0 #latvec1   
12 0.5 0 0.5 #latvec2   
13 0 0.5 0.5 #latvec3   
14   
15 ATOMIC_POSITIONS   
16 Cartesian   
17   
18 Si #label   
19 O #magnetism

```csv
20 2 #number of atoms
21 00 0 m 1 1 1 v 1.75205850628e-05 0.000155425594558 -3.99334763874e-05
220.241 0.2550.250999999m11 1 v-1.75205850628e-05-0.000155425594558 3.99334763874e-05
```

## 用HefeiNAMD进行非绝热分子动力学（NAMD）计算

Hefei NAMD 代码为 python 代码，在src 目录下，建议使用 python 3.9。需要的 python 库有 Numpy，Scipy,Numba，MPI4py （>= 3.1.3）。

## 设置NAMD 计算参数

在 Args.py 中设置NAMD 的各种参数，在 Args.py 的注释中有关于各种参数具体含义的详细介绍。

1 # NAMD parameter   
2 # manual input start   
3 dftdir= '/public/share/zhaojin/tuyy/abacus/sh/OUT.autotest1/'   
4 namddir = '../namd_test/' # output NAMD output file in namddir   
5 dt =1 # MD time step (fs)   
6 start_t = 1 # start MD step   
end_t  = 2000  # end MD step   
8 istart_t = 901  # isample start MD step   
9 iend_t = 1000  # isample end MD step   
10   
11 LCHARGE = True  # output atom projected charge density   
12 atom = [13,26]# atom number of all species (only needed in atomic basis)   
13 orbital = [27,13]# atomic orbital basis number (only needed in atomic basis)   
14 whichA = [0,13,14,15] # atom index for projected charge density (starts from 0)   
15   
16 LRANGE  = True # select range of band, change iband   
17 # if not given， LRECOMB specifies energy range   
18 # if LRECOMB not given， energy range: [O,nbands]   
19 LHOLE = True  # Hole/electron transfer   
20 dE = 2.0# initial energy from VBM/CBM (eV)   
21   
22 LPHASE  = True  # phase correction   
23   
24 TEMP = 300 # temperature in Kelvin   
25 NACTIME = 1000 # time for used NAC (i_end_t-state_t+NACTIME<nstep)   
26 NAMDTIME = 1000 # time for NAMD run   
27 NELM = 1000 # electron time step (per fs)   
28 NTRAJ = 5000 # SH trajectories number   
29   
30 LINTERP = 2 # hamiltonian interpolation algorithm 1,2,3   
31 LTA = False # Liouville-Trotter algorithm for small NELM   
32 LSH = 'FSSH' # run DISH， FSSH or DCSH   
33   
34 LREC0MB = False # consider electron-hole recombination   
35   
36 OutMode = 'text' # output file format 'numpy(.npy)' or 'text'   
37 # manual input end

## 读取ABACUS 的输出文件

运行 NAC.py 即可将 ABACUS 的输出文件转成 Hefei NAMD 需要的 NATXT和 EIGTXT文件。

在超算上计算的话，可以直接使用用 sub_scripts 目录下的 sub_nac 脚本提交任务，用户可在 sub_nac 中自行修改节点数等信息。

1 sbatch sub_nac

进行非绝热分子动力学计算

运行 SurfHop.py 即可进行NAMD 的计算。

在超算上计算的话，可以直接使用用 sub_scripts 目录下的sub_sh脚本提交任务，用户可在 sub_sh中自行修改节点数等信息。

1 sbatch sub_sh

输出文件中，SUPROP 文件中包含了电子驰豫能量和电子占据数随时间的演化，PSICT文件则包含波函数系数随时间的演化。可用 htps://github.com/QijingZheng/Hefei-NAMD/tree/master/scripts 中的脚本进行画图。

## 6.12.3结语

本篇主要介绍了如何利用ABACUS和HefeiNAMD进行非绝热分子动力学的计算，简单来说就是通过ABA-CUS 跑分子动力学得到每一步的波函数和哈密顿量等信息，HefeiNAMD再读入这些信息进行非绝热分子动力学的计算。关于非绝热分子动力学的详细介绍则可以查看Hefei NAMD的相关网站（见最上面的网站)，如有问题欢迎联系。

## 6.12.4 参考文献

[1] Zheng,Q.; Chu, W.; Zhao,C.; Zhang,L.; Guo,H.; Wang,Y.; Jiang,X.; Zhao,J. Ab initio nonadiabatic molecular dynamics investigations on the excited carriers in condensed mater systems.Wiley Interdiscip. Rev. Comput. Mol. Sci. 2019, 9, e1411.

## 6.13 Pyabacus 文档一：用户手册

## 6.13.1 简介

pyabacus 是密度泛函理论软件ABACUS（原子算筹）的Python 接口，提供了与 ABACUS 软件交互的 PythonAPI。

该项目基于 pybindl1 和 scikit-build-core 构建。

## 6.13.2 模块

pyabacus 当前提供以下模块：

io:pyabacus 中的输入/输出模块。

·Cell:用于单元结构的模块，将Python中的ModuleNAO与用户输入连接。

。 ModuleBase:基本数学函数模块。

。 ModuleNAO:数值原子轨道(NAO)模块。

。 hsoiver:用于求解哈密顿量的模块。

## 6.13.3安装

我们建议用户使用conda 进行安装，conda是一个开源的软件包管理系统和环境管理系统，可以方便的管理不同版本的Python 和软件包。用户可以按照以下步骤安装Pyabacus:

创建一个虚拟环境，并且激活虚拟环境 conda create -n pyabacus python=3.8& conda activate pyabacus下载ABACUS 软件

1 cd {your_working_directory}   
2 git clone htps://github.com/abacusmodeling/abacus-develop.git   
3 cd abacus-develop/python/pyabacus

安装 ABACUS 所需的依赖库 使用 pip install-v.安装 Pyabacus，或者使用 pip install.[test]安装 Pyabacus 及其测试环境的依赖库(使用 pip install-v.[test] -i https://pypi.tuna.tsinghua.edu.cn/simple 加速安装过程)。

## 6.13.4调用

用户可以通过以下代码调用Pyabacus 的接口：

```python
1 import pyabacus
2 s = pyabacus.ModuleBase.Sphbes()
3 s.sphbesj(1,0.0)
40.0
```

上面的代码调用了ABACUS 的 ModuleBase::Sphbes 模块。

目前，我们移植了下面几个模块：

·ModuleBase::Sphbes

ModuleBase::Integral

·hsolver::dav_subspace 在 Python 中调用这些模块的方法可见examples 文件夹内示例。

## 6.13.5示例

examples 文件夹中包含了一些Pyabacus 的示例脚本，用户可以通过这些示例脚本了解 Pyabacus 的使用方法。  
运行examples 文件夹中的 python vis_nao.py 以可视化数值原子轨道。

```shell
1 $ cd examples/
2 $ python vis_nao.py
```

运行 examples 文件夹中的 python ex_s_rotate.py 来检查 S矩阵。

```perl
1 $ cd examples/
2 $ python ex_s_rotate.py
3 norm(S_e3 - S_numer) = 3.341208104032616e-15
```

运行 examples 文件夹中的 python diago_matrix.py 来对角化一个矩阵

1 \$ cd examples/   
2 \$ python diago_matrix.py   
3 eigenvalues calculated by pyabacus: [-0.38440611 0.24221155 0.31593272 0.53144616 0.85155108   
4 1.06950155 1.11142051 1.12462152]   
5 eigenvalues calculated by scipy: [-0.38440611 0.24221155 0.31593272 0.53144616 0.85155108   
6 1.06950154 1.11142051 1.12462151]   
error:[9.26164700e-12 2.42959514e-10 2.96529468e-11 7.77933273e-12 7.53686002e-12 2.95628810e-09   
8 1.04678111e-09 7.79106313e-09]

## 6.13.6 测试

Pyabacus 使用 pytest 进行单元测试，用户可以在 abacus-develop/python/pyabacus/tests 目录下使用 pytest-v命令进行单元测试。

## 6.13.7结语

以上就是Pyabacus的一些基本安装和使用方法,如果读者对进一步交流感兴趣,欢迎登录ABACUS的github网站进行进一步的交流。

## 6.14 Pyabacus 文档二：HSolver 模块

## 6.14.1 简介

目前，pyabacus.hsolver 内实现了三种对角化算法的 python 化，分别是 hsolver.dav_subspace、hsolver.davidson与 hsolver.cg，其签名分别如下：

```python
def dav_subspace(
2 # 矩阵-向量组乘法函数，接收一个 numpy 数组，返回一个 numpy 数组
3 mvv_op: Callable[[NDArray [np.complex128]]， NDArray[np.complex128]],
4 # 对角化过程中的特征向量组初猜
5 init_v: NDArray[np.complex128],
6 #矩阵维度
7 dim: int,
8 # 要求的特征值个数，从最小的特征值开始，算法会求解 num_eigs 个特征值
9 num_eigs: int,
10 # 预处理向量
11 precondition: NDArray[np.float64],
12 # 基组允许容纳的向量个数 (dav_ndim * nband 个)
13 dav_ndim: int = 2,
14 #迭代误差
15 tol: float = 1e-2,
16 # 最大迭代次数
17 max_iter: int = 1000,
18 # 是否使用子空间函数
19 need_subspace: bool = False
20 ） -> Tuple[NDArray[np.float64]，NDArray[np.complex128]] :
21
22 # dav_subspace 会返回一个元组，分别为特征值数组与其对应的特征向量集合
```

23  
24 def dav_subspace(  
25 # 矩阵-向量组乘法函数，接收一个 numpy 数组，返回一个 numpy 数组  
26 mvv_op: Callable[[NDArray[np.complex128]]， NDArray[np.complex128]],  
27 # 对角化过程中的特征向量组初猜  
28 init_v: NDArray[np.complex128],  
29 #矩阵维度  
30 dim: int,  
31 # 要求的特征值个数，从最小的特征值开始，算法会求解 num_eigs 个特征值  
32 num_eigs: int,  
33 # 预处理向量  
34 precondition: NDArray[np.float64],  
35 # 基组允许容纳的向量个数 (dav_ndim \* nband 个)  
36 dav_ndim: int = 2,  
37 #迭代误差  
38 tol: float = 1e-2,  
39 #最大迭代次数  
40 max_iter: int = 1000,  
41 # 是否使用 PAW 方法  
42 use_paw: bool = False  
43 ） -> Tuple[NDArray[np.float64]， NDArray[np.complex128]]:  
44  
45 # davidson 会返回一个元组，分别为特征值数组与其对应的特征向量集合  
46  
47 def cg(  
48 # 矩阵-向量组乘法函数，接收一个 numpy 数组，返回一个 numpy 数组  
49 mvv_op: Callable[[NDArray[np.complex128]]， NDArray[np.complex128]],  
50 # 对角化过程中的特征向量组初猜  
51 init_v: NDArray[np.complex128],  
52 #矩阵维度  
53 dim: int,  
54 # 要求的特征值个数，从最小的特征值开始，算法会求解 num_eigs 个特征值  
55 num_eigs: int,  
56 # 预处理向量  
57 precondition: NDArray[np.float64],  
58 #迭代误差  
59 tol: float = 1e-2,  
60 #最大迭代次数  
61 max_iter: int = 1000,  
62 # 是否使用子空间函数  
63 need_subspace: bool = False,  
64 # 计算模式：自洽场/非自洽场  
65 scf_type: bool = False,  
66 #进程数  
67 nproc_in_pool: int = 1  
68 ） -> Tuple[NDArray[np.float64]， NDArray[np.complex128]]:  
69  
70#cg 会返回一个元组，分别为特征值数组与其对应的特征向量集合

## 6.14.2 安装 pyabacus

我们建议用户使用conda 进行安装，conda是一个开源的软件包管理系统和环境管理系统，可以方便的管理不同版本的Python 和软件包。用户可以按照以下步骤安装 pyabacus:

创建一个虚拟环境，并且激活虚拟环境 conda create -n pyabacus python=3.8&conda activate pyabacus

下载 ABACUS

1 cd {your_working_directory}   
2 git clone https://github.com/abacusmodeling/abacus-develop.git   
3 cd abacus-develop/python/pyabacus

安装 ABACUS 所需的依赖库使用 pip install-v.安装 pyabacus，或者使用 pip install.[test]安装 Pyabacus 及其测试环境的依赖库(使用 pip install-v .[test] -i https://pypi.tuna.tsinghua.edu.cn/simple 加速安装过程)。

## 6.14.3调用对角化算法

三种对角化方法签名几乎一致，只有微小的差别，调用方法也相同，下面统一用diago 指代。

在调用这两个函数前，首先我们需要读取一个矩阵，我们可以从 htps:/sparse.tamu.edu/PARSEC上下载，之后使用 scipy 读取，并且确定要求的特征值个数（示例中为8个)：

```python
1 h_mat = scipy.io.loadmat(mat_file)['Problem']['A'][0, 0]
2 dim = h_mat.shape[0]
3 num_eigs = 8
```

我们也可以用 numpy 生成一个对角占优的Hermitian 矩阵：

```python
1 n = 500
2 h_mat = np.random.rand(n,n)
3 h_mat = h_mat + h_mat.conj().T + np.diag(np.random.random(n))*10
```

之后，我们可以定义mvv_op算子：

```python
1 def mvv_op(x):
2 return h_mat.dot(x)
3 # 如果你愿意的话，也可以定义为一个 lambda 函数
4 # mvv_op = lambda x: h_mat.dot(x)
```

选取初猜vo:

```javascript
1 v0 = np.random.rand(nbasis, nband)
```

计算预处理子precond，由于这里我们读取的矩阵为对角占优的稀疏矩阵，所以我们可以取出矩阵的对角元组成新的矩阵D，并且计算D的逆作为矩阵的近似逆。

```python
1 diag_elem = h_mat.diagonal()
2 # 为了数值稳定，防止部分对角元过小（接近0）导致
3 # 其倒数为无穷大，我们将小于 1e-8 的点设置为 1e-8
4 diag_elem = np.where(np.abs(diag_elem) < 1e-8, 1e-8,diag_elem)
```

```python
5 precond = 1.0 / np.abs(diag_elem)
```

做完上述一系列准备工作后，我们即可调用pyabacus 的算法来计算特征值问题了！

```python
1 # 下面的 diago 可按需替换为 dav_subspace 或 davidson
2 e, v = diago(
3 mm_op,
4 v0，
5 nbasis,
6 nband,
precond,
8 dav_ndim=8，# cg 法无该参数
9 tol=1e-8,
10 max_iter=1000
11 ）
12
13 print(f'eigenvalues calculated by pyabacus is: \n'，e)
```

## 6.14.4结语

以上就是Pyabacus的一些基本安装和使用方法,如果读者对进一步交流感兴趣,欢迎登录ABACUS的github网站进行进一步的交流。

## 6.15Pyabacus文档三：开发者指南

## 6.15.1 简介

欢迎来到pyabacus 项目！pyabacus 是国产第一性原理软件 ABACUS（中文名原子算筹）的Python接口，它旨在为用户提供了便捷的Python API，使用户能够利用Python语言更灵活地实现电子结构计算并分析结果。本文档主要针对开发者撰写，以下是本文档的目录

## 。 项目结构

根目录CMake配置

模块CMake配置

·Python化上手指南

·Python 化指南

如何编写单元测试

. 如何参与开源贡献

## 。 资料汇总

读者可以参考该系列文档的前两个文档来熟悉pyabacus。如果读者希望做出贡献，本指南将帮助您了解项目结构、开发流程和代码贡献方式。

## 6.15.2 结构

## 目录结构

pyabacus项目的目录结构如下所示：

1 pyabacus/   
2 CMakeLists.txt   
3 src   
4 pyabacus   
5 {your_module}   
6 {interface}.py   
7 __init__.py   
8 {your_module}   
9 {your_code} .cpp   
10 CMakeLists.txt

项目采用 pybind11和 scikit-build-core 构建，借助CMake 构建工具链。因此，CMakeLists.txt 的配置是深人理解项目结构的关键。

## 根目录CMake配置

根目录下的CMakeLists.txt是pyabacus项目的主要配置文件。它设置了项目、查找必要的依赖项、配置构建选项，并为不同模块包含子目录。以下是文件中每部分的详细说明：

```cmake
1 cmake_minimum_required(VERSION 3.15...3.26)
2
3#项目设置
4 project(
5 \${SKBUILD_PROJECT_NAME}
6 VERSION \${SKBUILD_PROJECT_VERSION}
7 LANGUAGES CXX)
```

此部分设置项目名称、版本和使用的编程语言（本项目中为C++) 。项目名称和版本分别从SKBUILD_PROJECT_NAME 和 SKBUILD_PROJECT_VERSION变量中获取。

```cmake
1 # 查找 Python 和 pybind11
2 find_package(Python REQUIRED COMPONENTS Interpreter Development .Module)
3 find_package(pybind11 CONFIG REQUIRED)
```

该部分查找所需的Python 和 pybind11包。

```cmake
1 # 设置源路径
2 set(ABACUS_SOURCE_DIR "\${PROJECT_SOURCE_DIR}/../../sOurCe")
3 set(BASE_PATH "\${ABACUS_SOURCE_DIR}/module_base")
4 set(NAO_PATH "\${ABACUS_SOURCE_DIR}/module_basis/module_nao")
5 set(HSOLVER_PATH "\${ABACUS_SOURCE_DIR}/module_hsolver")
6 set(PSI_PATH "\${ABACUS_SOURCE_DIR}/module_psi")
set(ENABLE_LCAO ON)
8 list(APPEND CMAKE_MODULE_PATH "\${PROJECT_SOURCE_DIR}/../../cmake")
```

此部分设置了各种源路径和配置选项，定义了不同模块的路径并附加了自定义的CMake模块路径。

```cmake
#添加数学库
2 if(DEFINED ENV{MKLROOT} AND NOT DEFINED MKLROOT)
3 Set (MKLROOT "\$ENV{MKLROOT}")
4 endif()
if(MKLROOT)
set (MKL_INTERFACE lp64)
Set(ENABLE_MPI ON)
if (ENABLE_MPI)
find_package(MPI REQUIRED)
10 include_directories(\${MPI_CXX_INCLUDE_PATH})
11 endif()
12
13 set(USE_OPENMP ON)
14 if (USE_OPENMP)
15 find_package(OpenMP REQUIRED)
16 set (CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} \${OpenMP_CXX_FLAGS}")
add_link_options(\${OpenMP_CXX_LIBRARIES})
18 endif()
19 find_package(MKL REQUIRED)
20 add_definitions(-D__MKL)
21 include_directories(\${MKL_INCLUDE} \${MKL_INCLUDE}/fftw)
22
23 if(NOT ENABLE_DEEPKS)
24 list(APPEND math_libs IntelMKL: :MKL)
25 endif()
26
27 if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
28 list(APPEND math_libs -lifcore)
29 endif()
30 else()
31 find_package(FFTW3 REQUIRED)
32 add_compile_definitions(__FFTW3)
33 find_package(LAPACK REQUIRED)
34 include_directories(\${FFTW3_INCLUDE_DIRS})
35 list(APPEND math_libs FFTW3: :FFTW3 LAPACK: :LAPACK)
36
37 if (ENABLE_LCAO)
38 find_package(ScaLAPACK REQUIRED)
39 list(APPEND math_libs ScaLAPACK: :ScaLAPACK)
40 endif()
41 endif()
```

此部分配置数学库。它检查Intel数学核心库(MKL)的可用性并在可用时进行配置。如果MKL不可用，则改用FFTW3和LAPACK。同时，若启用MPI和OpenMP，也进行相应配置。

1 ＃ 添加包含目录   
2 include_directories(   
3 \\${BASE_PATH}

```shell
4 \${ABACUS_SOURCE_DIR}
5 \${ABACUS_SOURCE_DIR}/module_base/module_container
6 ）
```

此部分添加了项目所需的包含目录。

```cmake
1 # 添加基本库
2 Set(CMAKE_POSITION_INDEPENDENT_CODE ON)
3 # 添加 base 模块
4 set(BASE_BINARY_DIR "\${PROJECT_SOURCE_DIR}/build/base")
5 add_subdirectory(\${ABACUS_SOURCE_DIR}/module_base \${BASE_BINARY_DIR})
6 # 添加 parameter 模块
7set(PARAMETER_BINARY_DIR "\${PROJECT_SOURCE_DIR}/build/parameter")
8 add_subdirectory(\${ABACUS_SOURCE_DIR}/module_parameter \${PARAMETER_BINARY_DIR})
9 # 添加 orb 模块
10 set(ORB_BINARY_DIR "\${PROJECT_SOURCE_DIR}/build/orb")
11 add_subdirectory(\${ABACUS_SOURCE_DIR}/module_basis/module_ao \${ORB_BINARY_DIR})
```

此部分设置位置无关代码标志，并为 base、parameter、orb 模块添加子目录，指定这些模块的构建目录。

1 #设置 RPATH   
2 execute_process(   
3 C0MMAND "\\${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib;   
4 print(get_python_lib())"   
5 OUTPUT_VARIABLE PYTHON_SITE_PACKAGES   
6 OUTPUT_STRIP_TRAILING_WHITESPACE   
7 ）

此部分设置 Python 包目录的运行时搜索路径(RPATH)，使用Python命令获取 site-packages 路径并将其存储在 PYTHON_SITE_PACKAGES 变量中。

```cmake
1 # 将包名设置为 pyabacus
set(TARGET_PACK pyabacus)
3 Set(CMAKE_INSTALL_RPATH "\${PYTHON_SITE_PACKAGES}/\${TARGET_PACK}")
```

此部分将包名设置为 pyabacus，并配置安装 RPATH以包含Python site-packages 目录。

```cmake
1 # 为子模块添加子目录
2 add_subdirectory(\${PROJECT_SOURCE_DIR}/src/hsolver)
3 add_subdirectory(\${PROJECT_SOURCE_DIR}/src/ModuleBase)
4 add_subdirectory(\${PROJECT_SOURCE_DIR}/src/ModuleNAO)
```

此部分为模块添加子目录，每个子目录包含自己的CMakeLists.txt以进行进一步配置。

通过遵循此结构，CMakeLists.txt文件确保了所有必要的依赖项被找到、配置并包含在构建过程中。它还设置了项目环境并包含不同组件的子模块。

## 模块 CMake配置

以下示例为模块（如 pyabacus.hsolver）的CMakeLists.txt文件：

```cmake
1 # 添加 diago 动态链接库
2 list (APPEND _diago
3 \${HSOLVER_PATH}/diago_dav_subspace.cpp
\${HSOLVER_PATH}/diago_david.cpp
5 \${HSOLVER_PATH}/diag_const_nums.cpp
6 \${HSOLVER_PATH}/diago_iter_assist.cpp
\${HSOLVER_PATH}/kernels/dngvd_op.cpp
8 \${HSOLVER_PATH}/kernels/math_kernel_op.cpp
9 \${BASE_PATH}/kernels/math_op.cpp
10 \${BASE_PATH}/module_device/device.cpp
11 \${BASE_PATH}/module_device/memory_op.cpp
12 \${PSI_PATH}/psi.cpp
13）
14 add_library(diagopack SHARED \${_diago})
15 target_link_libraries(diagopack
16 base
17 parameter
18 container
19 orb
20 \${math_libs}
21 \${OpenBLAS_LIBRARIES}
22 \${LAPACK_LIBRARIES}
23）
24
25 list(APPEND pymodule_hsolver
26 \${PROJECT_S0URCE_DIR}/src/hsolver/py_hsolver.cpp
27）
28
29 # 使用 pybind11 添加 Python 模块
30 pybind11_add_module(_hsolver_pack MODULE \${pymodule_hsolver})
31 # 将依赖项和 pybind11 库链接到模块
32 target_link_libraries(_hsolver_pack PRIVATE pybind11::headers diagopack)
33 target_compile_definitions(_hsolver_pack PRIVATE VERSION_INFO=\${PROJECT_VERSION})
34
35 set_target_properties(diagopack PROPERTIES INSTALL
```

## Python化上手指南

·Pybind 简单教程（外部链接，感觉写的比较简单，可以直接阅读官方文档）里边有我入门pybind11时做的一些笔记，可以理解为官方文档的部分摘抄和学习

。 阅读 pybindl1 with cmake，了解如何搭配cmake 使用 pybind11

·我们的项目使用 pybindl1+cmake+scikit-build-core 构建，所以可以下载 scikit-build-core template 仓库，搞懂里边的代码，尝试跑一个demo 出来

有任何与 scikit-build-core 相关的问题可以查阅 scikit-build-core 文档

·阅读 pyabacus README，简单了解 pyabacus 目前的模块结构，尝试按照 README 编译 pyabacus 并跑通（可

以运行 python/pyabacus/examples 内的 python 脚本）

·阅读 pyabacus How to contribute，同步阅读 pyabacus 下各文件夹内 CMakelists，理解它们之间的关系。本文档中给出了该部分的中文翻译，可对照查看。

## 。 同步地，可以看一下 src/hsolver 内的代码，其结构如下

{method}.hpp 为对应算法的头文件，头文件中将 hsolver 内对应算法进行了一定的包装，加人了 pybind 的一些头文件以方便调用py_hsolver.cpp为pybind11绑定，将三个方法对应的类绑定到 python./pyabacus/hsolver内的两个py文件将类的细节隐藏，用函数在外层抽象，将三个算法以函数的形式暴露给 python

了解了上面几点，你大概就了解了如何 Python化ABACUS 的一个模块，剩下需要搞懂的是

·你要 Python化的模块，按照面向对象的思想，你可以只搞懂对应模块的头文件中暴露出来的 public方法的接□

但是，如果想要优化/防止python 化过度影响效率，最好要搞懂其原理作为高性能计算软件，我们时常与裸指针/底层数据打交道，但是我们处理的数据通常是矩阵or张量，所以一定要从源码中弄清楚数据的存储方式：行优先/列优先

不要想当然，作为有着”历史积累”的软件，不同模块可能使用不用方式存储

·例如：davidson 算法与dav_subspace 中的裸指针采用列优先方式，与 fortran一致，while CG算法中的 tensor的底层数据是行优先方式，与C++习惯及NumPy中一致

## 。 如何编写单元测试

可以阅读tests文件夹内的代码，同时查阅pytest文档 提交PR前一定要做好单元测试，确保功能没有问题！看到一片绿色写着PASS 时候是很有成就感的

对上面问题有了基本的了解以后，我们就可以进行开发了！祝你在开发的过程中好运

## 6.15.3 资料汇总

Pybind 简单教程  
pybind11 with cmake  
scikit-build-core template  
scikit-build-core 文档  
pyabacus README  
pyabacus How to contribute  
pytest 文档  
寻找在GitHub 上参与开源项目的方法  
GitHub Issues 快速入门  
https://zhuanlan.zhihu.com/p/697280920

## 6.15.4结语

以上就是Pyabacus 的一些基本安装和使用方法,如果读者对进一步交流感兴趣,欢迎登录ABACUS的github网站进行进一步的交流。

## 第7章开发者规范

## 7.1 ABACUS 的Github 仓库Issues 处理流程

## 7.1.1 Github 的 Issues 机制

## 简介Github 的 Issues

GitHub 上的Issues 机制是一个用于跟踪和讨论项目相关问题、需求、任务和改进的功能。它为开发者们提供了一个集中式的平台，使得项目的参与者们能够跟踪项目进展、报告错误、提出新功能、讨论改进方向等。

## Issues对ABACUS软件的作用

ABACUS（原子算筹）是一款国产密度泛函理论（Density Functional Theory，简称DFT）软件，其涉及到的功能繁多，仅2023年上半年统计到的输入参数就有三百多个，应用的对象也很广，在材料、物理、化学、能源和工程等多个领域都有应用。因此，常见的问题是开发者开发出的功能不能包含所有可能的应用对象，因此软件在某些特定场合出现效率不高甚至某个功能无法运行的情况是时有发生的。另外一方面，ABACUS项目参与的开发人员广泛，不同开发者对同一个软件代码进行修改，容易引发新的bug。因此，建立一个良好的用户和开发者，乃至开发者之间互动交流的平台，对于发现软件问题，不断改善软件质量和提高稳定性易用性方面具有重要作用。Github上的Isses 机制则承担起了这一部分功能。因此，将 Issues 机制建立好是不断提升ABACUS软件，使其快速迭代的重要措施。

## Issues 功能介绍

·创建Issue:项目的任何参与者都可以新建一个Isse。创建Issue时，需要填写标题、描述（可以使用 Markdown语法格式化描述内容)、分配处理人（Assignees，可选）、添加分类标签(Labels，可选)、关联项目看板（Projects，可选）和设置里程碑（Milestones，可选）。

·标签(Labels)：标签是用来对Issues 进行分类和过滤的。项目维护者可以创建自定义标签,例如:bug、feature、documentation、enhancement 等。这有助于组织和优先处理不同类型的问题。

·里程碑（Milestones）：里程碑用于对 Isues 进行分阶段管理和规划。项目维护者可以创建里程碑并为其分配截止日期，这样有助于按照计划进行项目开发。

·指派（Assignees）：在Issue上指派处理人意味着将该Issue的负责人分配给特定的项目参与者。这有助于清楚地了解谁负责解决某个问题。

·评论和讨论：项目的参与者们可以在 Issue下面发表评论，进行讨论。评论支持 Markdown 语法，因此可以方便地插入代码、图片和链接等。

·引用和关联Issue:在评论和提交（Commit）中，可以通过输入类似 #issue 编号的格式引用其他Isue，这有助于将讨论串联起来。此外，还可以使用关键字（如：close、fix、resolve等）在提交信息中自动关闭相关Issue。

·通知和订阅:当您参与某个Issue时（如创建、评论或被指派)，如果您希望及时收到回复，建议您关注是否会收到邮件通知 (关注一下Github 的功能)

·关闭和重开Issue:当某个问题被解决或者不再适用时，项目维护者或 Issue创建者可以关闭Issue。如果后续发现问题没有解决或者需要重新讨论，可以重新打开Issue。

## 7.1.2 ABACUS项目的Issues标签分类

提交Issues 会让提交者（提交者可以是用户或者开发者）选择标签，以下是ABACUS开发者团队推荐的标签及其使用场景

## 分类：

Bug Report（开发者/用户)：用于报告代码问题，例如崩溃、错误或意外行为。

·Code Quality（开发者)：用于报告与代码质量相关的问题，例如代码不规范、设计不良、全局类或去全局变量使用不当、或者可维护性问题，开发者会用重构来解决相关问题。

·Docs (开发者/用户)：用于报告与软件文档相关的问题，例如缺少或不完整的文档，或难以理解的文档。

Feature Request（开发者/用户)：用于请求新功能或对现有功能进行增强。

·Performance（开发者/用户)：用于报告与用户角度下软件性能相关的问题，例如响应时间慢、资源使用率高或可扩展性问题。

。 Questions（开发者/用户)：软件安装、软件易用性等用户或开发者需要帮助解决的问题。

·Tests（开发者)：用于报告与测试相关的问题，例如测试失败、缺少或不完整的测试或测试自动化问题

## 优先级：

·Priority High：通常指的是需要尽快解决的紧急问题，如CI不通过，新PR全部报错

Priority Medium：重要且需要近期解决的问题，如...

·Priority Low：可能是一些不紧急但需要解决的问题，如需求调整、性能优化

专题:

·DFTU,EXX,big system 等用于记录一些专题类Issues，以方便开发者系统解决同一个主题的Issues

## 7.1.3 Issues 处理流程

## 介绍

提交的Issues 会在24 小时内得到开发者团队的回复。当前Issues 处理的闭环方式有：

·提交PR 修改代码，解决对应Issues，之后关闭相应的Issue。

·ISSUE历史大于两周以上，且ISSUE 提出者不对有效回应做回应，则提醒用户后关闭

·属于Discussion 机制而不属于Issues 的范围，会转移到Github 的Discussion 模块：例如用户的想法，用户的建议，用户对社区的问题等以用户的满意程度为指标的问题

## 创建一个ISSUE

有多种方法可以创建ISSUE，您可以选择最方便的方法来。本例将使用GitHubUI。有关其他创建ISSUE的方法的更多信息，请参考Github文档“创建ISSUE”。以下是具体步骤：

1.在 GitHub 上，导航到 ABACUS 仓库的主页。htps:/github.com/deepmodeling/abacus-develop.git

2.在 ABACUS仓库名称下面，单击“Issues”。

<!-- image-->

3.单击“New issue”。

---

## 第 301–320 页

<!-- image-->

4.选择一个最适合您的 ISSUE的一个标签模版，点击“Get started”。如果 ABACUS 提供的默认模版不符合您的ISSUE 类型，可以选择“Openablankissue”。

<!-- image-->

## 填人ISSUE信息

给您的 ISSUE一个描述性的标题。标题应该一眼就能传达问题的内容。添加一个解释 ISSUE 目的的描述，包括任何可能有助于解决问题的细节。例如，如果这是一个bug报告，描述重现bug的步骤、预期结果和实际结果。您可以使用Markdown 添加格式、链接、表情等。有关更多信息，可参阅“在GitHub 上撰写”。

以下是 ABACUS 的 Bug report 类型 ISSUE 的模版，您可以根据 ISSUE 模版示例来填人ISSUE 信息。其中带“\*”的项是必填项。

## Issue: Bug Report

ForsersorelorstrtoggsshosptedfIftstht choose a different type.

<!-- image-->

## ISSUE的任务列表

请注意“Task list forIssue attackers”是为了帮助开发者检查ISSUE解决的进度,并不是要求您选择解决ISSUE需要完成哪些工作。所以您只需要提供足够详细的信息以帮助开发者来解决ISSUE。如果您提供的信息已经完成其中的一些任务列表，可以在相应的任务前打对勾，但是如果没有可以不用关心“Task list forIssue attackers”。当然，了解解决一个ISSUE所需要经过的一般步骤，有助于您了解 ABACUS代码的开发过程。如果您想通过自定义的任务列表给开发者一些提示，可以参考Github文档“About task lists”。

<!-- image-->  
③ Remember,contributions to this repository should follw its contributing guidelines.

## 添加ISSUE标签

如果您选择ISSUE 模版，创建完ISSUE之后会自然有相应的标签出现，如：

<!-- image-->

此外，您也可以参考ABACUS 的专题标签（例如DFTU,EXX等）添加相应的标签，或者通过“Edit labels”创建新标签：

<!-- image-->

## 提交ISSUE

点击“Submit new issue”以创建您的ISSUE。您可以在创建问题后编辑上述任何字段。您的ISSUE 有一个唯一的URL，您可以与团队成员分享，或在其他 ISSUE 或PullRequat中引用。

<!-- image-->

## 与开发者交流

创建ISSUE后，通过添加评论与开发者或其他协作者交流对话。您可以键人@邀请协作者或团队关注评论。

<!-- image-->  
③ Remember, contributions to this repository should follow its contributing guidelines.

要在同一仓库中可以链接相关问题，您可以键入#，后跟ISSUE标号或部分标题，然后单击要链接的问题。有关更多信息，请参见““在GitHub上撰写””。

<!-- image-->

## 查看ISSUE 解决进度

查看ISSUE解决有3种方式，如下图所示：

可以通过“task list”的完成进度查看ISSUE 的解决进度

新建的 ISSUE 会被自动加入ABACUS ISSUEs项目，开发者通过讨论之后会在其中加人"Iteration”解决周期，"Priority”优先级，"Colaborators”开发者单位，"Project”是否纳入近期的开发者项目等信息，您也可以在这里看到ISSUEs 的进展。

Assignees 是解决ISSUE 的开发者，需要ABACUS 管理员进行分诊。

<!-- image-->

## 关闭ISSUE

以第1步和第2步同样的方式找到想要关闭的 ISSUE。点击“Close issue”，关闭 ISSUE。

<!-- image-->

如果要添加关闭ISSUE的原因,可以在“Leave acomment”处添加关闭原因,然后点击“Close with comment”,关闭ISSUE。

<!-- image-->

## 7.2 ABACUS开源项目C++代码规范

## 7.2.1 前言

让任何开发者都可以快速读懂别人的代码，这点对于ABACUS项目很重要。此外，使代码易于管理的方法之一是加强代码一致性，这需要从代码规范开始着手。

## 7.2.2命名约定

## 普通变量的命名统一是小写，并使用下划线命名法

以下是四种命名方法：下划线命名法、匈牙利命名法、驼峰式命名法、帕斯卡命名法

1 <strong>string table_name；</strong>// 下划线命名法（推荐）-用下划线区分单词（推荐使用）  
2 string sTableName；// 匈牙利命名法－前缀字母用变量类型缩写，单词首字母大写  
3 string tableName；// 驼峰命名法 - 混合大小写 (不推荐使用）  
4 string TableName；// 帕斯卡命名法 － 每个单词首字母大写 (不推荐使用)

下划线命名法：推荐使用

匈牙利命名法：目前ABACUS中有些变量（例如指针)，首字母为数据类型（指针为p)，可以考虑用这种方式命名，但不推荐用大写字母

驼峰命名法/帕斯卡命名法：目前ABACUS 大部分的代码不是用这种命名方式的，为了代码风格统一，所以不推荐使用

## 函数命名

建议使用下划线命名法且所有字母全部小写，建议小于18个字符

1 void calculate_area() {   
2 // function body   
3

## private的类数据成员在最后以下划线结尾

private 的类数据成员和普通变量的命名方式一样，但要在最后以下划线结尾，以区分自己是private 的类数据成员。

1 class TableInfo  
2{  
3 ：  
4 4 private:  
5 string table_name_； // 好- 后加下划线  
6 string tablename_； // 好  
7 static Pool<TableInfo>\* pool_； // 好  
8 }；

## 只有全局变量用全大写命名

在 ABACUS 中，例如NBANDS（能带数)，NLOCAL（局域轨道数目）这种全大写的变量名是全局变量，其它的不建议用大写字母给变量命名。另外，ABACUS 将通过重构或者删除的方式逐步淘汰全局变量，因此不建议增加新的全局变量。

## 7.2.3关于头文件

只有在真正需要使用某个库/头文件时才 #include 它们

不需要的 #include 删除。

## #include避免使用快捷目录.(当前目录)或.. (上级目录)

项目内头文件应按照项目源代码目录的树结构排列，避免使用UNIX特殊的快捷目录.(当前目录)或..（上级目录)。例如

source/module_hsolver/diago_cg.h文件是这样引用module_base 的头文件:

1 #include "diagh.h"   
2 #include "module_base/complexmatrix.h"

## #include 顺序 (clang-format可以自动)

使用标准的头文件包含顺序，这样可以可增强可读性，避免隐藏依赖。

foo.cpp 中包含头文件的次序如下：

与源文件对应的头文件:dir2/foo2.h(这种优先的顺序排序保证当 dir2/foo2.h遗漏某些必要的库时,dir/foo.cpp的构建会立刻中止)

C系统文件  
C++系统文件  
其他库的.h 文件  
本项目内.h 文件

## 头文件需要 #define 保护

```c
1 #ifndef FOO_BAR_BAZ_H_
2 #define FOO_BAR_BAZ_H_
3 ：
4 #endif // FOO_BAR_BAZ_H_
```

## 避免使用前置声明

使用 #include 包含需要的头文件，尽量避免使用前置声明(见以下例子解释什么是前置声明)。

1 /／ 什么是前置声明？   
2 // MyClassB.h   
3   
4 // Bad: Overuse of forward declarations   
5 class MyClassB;   
6

```cpp
7 class MyClassA {
8 public:
9 void DoSomething(MyClassB* obj_b);
10 0}；
11
12 void MyClassA::DoSomething(MyClassB* obj_b) {
13 / ·
14}
```

在这段代码中，MyClassB被声明为一个类，但没有给出其定义。这被称为前置声明（Forward Declaration)，它告诉编译器MyClassB是一个存在的类，但不提供该类的详细信息。

## 当函数只有10行甚至更少时，才将其定义为内联函数

只有当函数只有10行甚至更少时才将其定义为内联函数（关键字inline）。

## 7.2.4关于类（Class)

先声明类的成员变量，再声类的成员函数

将 public 部分放在最前，之后是protected 数据，最后是private

## 不使用运算符重载

若要使用，建议提交issue 讨论

## 什么时候用struct ?

只有数据成员、没有成员函数时可以用struct

## 类型转换使用 static_cast<>(

建议使用C++ 的类型转换，如 static_cast<>(。

不要使用 inty=(int)x 或 inty=int(x)等转换方式。

用 static_cast 替代C 风格的值转换,或某个类指针需要明确的向上转换为父类指针时。

用 const_cast 去掉const 限定符。

用reinterpret_cast指针类型和整型或其它指针之间进行不安全的相互转换。仅在你对所做一切了然于心时使用。

## 继承采用public继承

如果你想使用私有继承，你应该替换成把基类的实例作为成员对象的方式去定义类。

## 不过度使用实现继承，可以考虑组合

例如，如下案例：

1 class Engine {  
2 public:  
3 void start(){ /\* 启动引擎 \*/}  
4}；

```cpp
5
6 class Car {
7 public:
8 Car(const std::string& n, Engine& e) : name_(n)，engine_(e) {}
9 void drive() { engine_.start(); std::cout << name_ << " is driving\n"; }
10
11 1 private:
12 std::string name_;
13 Engine& engine_;
14 }；
15
16 int main() {
17 Engine engine;
18 Car car("mycar",engine);
19 car.drive();
20 return 0;
21 }
```

## 7.2.5关于函数（Function)

## 函数传参：不变参数用const和&

函数的输入参数与输出参数:

在一个函数中，不变的量，我们可以看作是函数的输人参数；变化的量，我们可以看作是函数的输出参数。在输入参数中可以选择constT\*（指向常量对象的指针，不能通过这个指针来修改其指向的对象的值。然而，你可以改变指针本身的值)。

也可以使用constT&（不能通过这个引用来修改其引用的对象的值，在其生命周期内不能重新引用另一个对象)。

所以，建议使用constT&，若要使用constT\*，则应给出相应的理由，否则会使读者感到迷惑。

```c
1 void Foo(const string &in, string *out);
2 // 输入参数:in (const + 引用&)
3//输出参数：out (指针变量)
```

## 函数传参：会变的参数用指针

什么是引用参数:

在C 中,如果函数需要修改输入变量的值,参数必须为指针,如 int foo(int \*pval)。

在 C++ 中,函数还可以声明为引用参数:int foo(int&val)。

引用参数在语法上是值变量却拥有指针的语义 (变量可以被改变！)。

## 每个函数不超过50行

建议编写简短，凝练的函数，有特殊情况的除外。

函数返回值（return的值）多使用值返回和引用返回，避免使用指针返回

## 7.2.6关于作用域

## 变量要初始化

无初始化的变量可能会引起结果不稳定（例如出现随机数)，因此建议养成习惯，对所有变量的值要初始化，见下面的例子：

1 int i;  
2 i= f()；// 坏—初始化和声明分离  
3  
4 int j= f()；// 好—初始化时声明

1 vector<int> v； // 坏—初始化和声明分离  
2 v.push_back(1);  
3 v.push_back(2);  
4  
5 vector<int>ν = {1，2}；// 好—初始化时声明

## 将局部变量置于最小作用域

在if,while 和 for 语句中：

变量不是对象，则变量应当在内部声明与初始化，这样子这些变量的作用域就被限制在这些语句中了，举例而言：

1 while (const char\* p = strchr(str,'/'))   
2 {   
3 str = p + 1;   
4}

如果变量是一个对象，每次进入作用域都要调用其构造函数,每次退出作用域都要调用其析构函数。这会导致效率降低。建议将构造函数调用次数减少以提高程序效率。

1 Foo f；// 构造函数和析构函数只调用 1 次  
2 for (int i = 0; i < 1000000; ++i)  
3{  
4 f.DoSomething(i) ;  
5}  
6  
7/／低效的实现  
8 for (int i = 0; i < 1000000; ++i)  
9{  
10 Foo f；// 构造函数和析构函数分别调用 1000000 次！  
11 f.DoSomething(i);  
12}

## 仅在局部作用域使用using namespace

使用using namespace 语句来引入命名空间中的所有名称可能会导致名称冲突。因此，建议在需要时使用它并仅在局部作用域内使用（例如只在调用的时候使用)。

1 /／ 建议不在程序开头使用，而是在具体用到 std 库的函数内使用该语句  
2 using namespace std;

## 用全局函数要加命名空间

以下是两种建议的方式，或者用类，或者用namespace

1 /／类的静态成员函数  
2 class MyMath {  
3 public:  
4 static int add(int a,int b){  
5 return a + b;  
6 }  
7  
8 static int sub(int a,int b) {  
9 return a - b;  
10 }  
11 }；  
12  
13 //or  
14 /／命名空间内的非成员函数  
15 namespace my_math {  
16 int add(int a,int b){  
17 return a + b;  
18 }  
19  
20 int sub(int a,int b){  
21 return a - b;  
22 }  
23 }

## 优先使用命名空间的非成员函数

能使用命名空间的非成员函数，就不用类的静态成员函数。

```c
1 /／使用命名空间内的非成员函数
2 namespace my_math {
3 int add(int a,int b){
4 return a + b;
5 }
6
7 int sub(int a,int b) {
8 return a - b;
9 }
```

10}  
11  
12 int main() {  
13 int x = 3, y = 5;  
14 int z = my_math::add(x，y)；// 调用 my_math 命名空间中的函数  
15 return 0;  
16}

```cpp
1 /／不要使用类的静态成员函数模拟命名空间
2 class MyMath {
3 public:
4 static int add(int a,int b){
5 return a + b;
6 }
7
8 static int sub(int a,int b){
9 return a - b;
10 }
11 }；
12
13 int main() {
14 int x = 3, y = 5;
15 int z = MyMath::add(x，y)；// 调用 MyMath 静态方法
16 return 0;
17}
```

## 7.2.7 其他 C++ 特性

## 每个代码文件不超过500 行

太长的代码阅读理解和维护的成本都太高，因此不建议一个文件太长。如果有文件超过 500行，建议重构，把对象进一步的划分。

## 禁止用C++11之后版本的语法

目前的主要考虑是用新语法会使得编译器编译成功的概率降低，另外提高开发者的开发门槛。因此，我们规定不能使用C++11之后的语法。

## 多用前置自增(++i)

对于迭代器和其他模板对象使用前缀形式(++i)的自增，自减运算符。

不考虑返回值的话，前置自增(++i)通常要比后置自增(i++)效率更高

## 尽可能用 sizeof(a)代替 sizeof(int)

这里a是一个参数名

原因：当代码中变量类型改变时会自动更新。

## 多使用列表初始化

C++11中，该特性得到进一步的推广，任何对象类型都可以被列表初始化。示范如下：

1 // Vector 接收了一个初始化列表。  
2 /／ 不考虑细节上的微妙差别，大致上相同。  
3 /／ 可以任选其一。  
4 vector<string> v{"foo"， "bar"};  
5 vector<string> v = {"foo"，"bar"};  
6  
7/／ 可以配合 new 一起用。  
8 auto p = new vector<string>{"foo"， "bar"};  
9  
10 // map 接收了一些 pair，列表初始化大显神威。  
11 map<int，string> m = {{1，"one"}，{2，"2"}};  
12  
13 /／ 初始化列表也可以用在返回类型上的隐式转换。  
14 vector<int> test_function() { return {1，2，3}；}  
15  
16 /／ 初始化列表可迭代。  
17 for (int i : {-1，-2，-3}){}  
18  
19 //在函数调用里用列表初始化。  
20 void TestFunction2(vector<int> v) {}  
21 TestFunction2({1，2，3});

## 初始化时：整数用0，实数用0.0，指针用nullptr，字符(串)用'0’

C++11引入了一个新的关键字nullptr，用来表示空指针。相对于传统的NULL或0，nullptr 更加明确、类型安全。使用nulptr可以避免一些潜在的编程错误，比如将整数值误传给函数，导致出现不可预期的行为。

因此，建议在C++11及以上的版本中使用nullptr 来表示空指针。

## 少用auto，使用前需对auto有更全面的了解

auto 是C++11引入的关键字，它可以让编译器自动推导出变量的类型。之后，C++14和C++17对auto 的使用也有了一些扩展和改进。

C++11：auto只能用于定义局部变量，并且必须初始化。例如：

1 auto i = 42；// 推导出 i 的类型为 int  
2 auto f= 3.14f；// 推导出 f 的类型为 float  
3 auto s = "hello"；// 推导出 s 的类型为 const char\*

C++14：auto 可以用于定义函数返回值类型，使得函数定义更加简洁。例如：

1 auto add(int x，int y）{ return x + y;} // 推导出返回类型为 int  
2 auto divide(double x，double y） { return x ／ y; } // 推导出返回类型为 double

C++17：auto 进一步扩展为auto 和模板结合使用时，可以直接指定模板类型参数，从而实现更加灵活的类型推导。例如：

```cpp
1 std::vector<int> v{1, 2, 3};
2 auto it = v.begin(); // 推导出 it 的类型为 std::vector<int>::iterator
3 auto [first，second] = std:make_pair(1，3.14)；// 使用结构化绑定和 auto 推导出 first 和 second 的类型
```

但是大家注意，在abacus 中，我们只支持C++11的标准，C++14/17语法是不接受的。

## auto 和for的混合使用时注意事项

在C++11中，auto和 for循环的结合使用已成为一种常见的编程范式，它可以让代码更加简洁、易读，并且减少了手动指定类型的错误。

```c
1 int arr[] = {1,2,3};
2
3 for (auto i：arr） // 相当于复制
4{
5 std:cout << i << " ";
6}
7
8 for(const auto& i :arr)// 常量引用
9{
10 std:cout << i << " ";
11}
12
13 for（auto&i ：arr）// 能够实现对元素的直接修改。
14{
15 i *= 2;
16}
```

## constexpr替代宏定义和const常量

在C++11里，用constexpr来定义真正的常量，或实现常量初始化。(真正的常量在编译时和运行时都不变)

```c
#define PI 3.14159 // PI 是一个宏定义常量，它不会进行类型检查，容易出错；
const double kE = 2.71828；// kE 是一个 const 常量，它不能用于编译期计算。
3
4 constexpr double kGravity = 9.8;
```

constexpr可以替代宏定义和const常量的主要原因是：

类型安全：使用constexpr定义的常量会进行类型检查，避免了宏定义可能带来的类型错误，同时也比const常量更加严格。

编译时计算：constexpr声明的变量或函数在编译时就可以被求值，而不需要在运行时计算。这比宏定义和const常量更高效，尤其是在需要多次使用同一个值的情况下。

更好的可读性和可维护性：使用constexpr可以使代码更加清晰易懂，减少了宏定义可能导致的代码混乱问题。同时，由于constexpr可以使用函数、类等C++ 语言特性，因此更加灵活，对于复杂的计算也更容易维护和修改。

因此，在C++11及以上的版本中，建议使用constexpr来替代宏定义和const 常量，以提高代码的可读性、可维护性和效率。

## 7.2.8关于ABACUS中常用的关键词缩写

有些名字很长，我们希望尽量言简意赅的表达出一些关键词的意思。原则是一般3-5个字母的范围下尽量说清楚一个变量的含义。这些统一的命名会出现在函数名或者变量名里。

## 两个字符

pw：代表plane wave平面波

op：代表具有 multi-device 和 multi-precision 支持的算子（operator），和 Operator 模块含义不同

## 三个字符

fft：快速傅里叶变换

kpt：布里渊区kpoint的缩写

nao：代表 numerical atomic orbitals（nao 经常用来表示 number of atomic orbitals，不知道会不会混）

orb：orbital，轨道

hmt：代表 hamilt 或者 hamiltonian

pot：代表 potential

chg：代表charge

den：代表density（电荷密度尽量都用chg）

scf：代表自洽迭代 self consistent field

thr：代表 threshold

tab：代表table

kin：代表kinetic，动能的

cal:代表calculate

opt：代表optimize

gen：代表 generate

## 四个字符

iter：代表 iteration

init：代表初始化initializaiton

read:读入

stru：代表 structure

veff：代表有效势

vloc：代表局域势

## 7.2.9 Reference

Google C++ Style Guide

## 7.3 ABACUS 注释规范：Doxygen 人门 (c++)

## 7.3.1 简介

开源密度泛函理论软件ABACUS目前由众多社区开发者共同开发，因此有一个统一的代码注释规范是非常重要的，本文主要推荐以Doxygen工具为辅助的代码注释格式。

## 什么是Doxygen

Doxygen 是一款支持包括C/C++,Python,Fortran 等多种语言的静态程序分析与文档生成工具。通过搜集并整理代码中的信息，包括但不限于

函数的调用/被调用

类的成员以及继承、协作关系

文件夹之间的依赖关系

头文件 include关系

Doxygen不仅能自动生成包含以上信息的图表与列表，也能将代码中采用特定格式的注释附到相应位置，最终生成内容翔实的文档。在适当的 Doxygen 配置与良好的代码注释下，Doxygen 自动生成的文档能为开发者提供诸多便利，降低阅读代码的难度，从而促进开发者社区的建设。

## 功能与文档范例

Doxygen可以生成某个函数的调用与被调用图(call&caler graph)。以开源密度泛函理论软件 ABACUS 代码为例，其中负责计算三次样条插值的函数 ModuleBase:CubicSpline::eval，doxygen可将其调用关系自动可视化为如下图表:

<!-- image-->

Doxygen可以生成类的协作图（collaboration diagram)。下图展示了container:Tensor 的成员以及成员的成员之间的关系：

<!-- image-->

Doxygen 可以生成文件夹之间的依赖关系图。下图展示了source/module_basis 及其内部成员文件夹与其他文件夹的耦合。

<!-- image-->

Doxygen 可以生成文件夹之间的依赖关系图。下图展示了 source/module_basis 及其内部成员文件夹与其他文件夹的耦合。

<!-- image-->

Doxygen 可以将头文件的include关系可视化成图表(include dependency graph)。下图（部分地）展示了 mod-ule_basis/module_nao/numerical_radial.h包含与被包含的情况：

<!-- image-->  
大量项目使用 Doxygen 生成文档(https://www.doxygen.nl/projects.html)。对科学计算开发者而言一个著名的例子是LAPACK 的在线文档 https://www.netlib.org/lapack/explore-html/

<!-- image-->

## 7.3.2如何使用Doxygen

Doxygen 可以自动识别代码逻辑并生成以上例子中的函数调用、类的继承与协作、文件夹依赖以及头文件依赖等图表。但是，对理想的开发者文档而言代码注释中的信息亦不可或缺。在最终生成的文档中，除以上自动生成的图表外，Doxygen会将类的成员与函数一一列举，并识别代码中特定格式的注释后将其附在对应位置。

## 如何写能被Doxygen识别的注释

通常开发者会希望一部分精心撰写的注释——比如类的功能介绍或者函数参数的含义——出现在开发者文档里，而不希望文档充斥着临时变量的描述或某一行意义的解释。Doxygen充分考虑了这一点。事实上，普通的注释均会被Doxygen 忽略；只有特定格式的注释会被 Doxygen 识别。下面的代码是一个简单的例子。尽管其中各处都附上了注释，但这些均不会出现在最终生成的文档中：

1 // A class that represents a circle.   
2 // This toy class will be used to demonstrate the usage of Doxygen.   
3 class Circle {   
4 public:   
5 // Initializes a Circle object with a given radius.   
6 // A unit circle is assumed by default.   
7 Circle(double r = 1.0);   
8   
9 // Rescales a Circle object.   
10 // The radius of the given Circle object is multiplied by the given factor.   
11 static void rescale(Circle\* circ, // pointer to the Circle object   
12 double factor // radius rescaling factor   
13 ）；   
14   
15 private:   
16 double r_; // radius of the Circle object   
17 static constexpr double PI = 3.14159265358979324;

---

## 第 321–340 页

以上代码生成的文档如下。Doxygen 会自动将成员变量和函数一一列举，但由于注释不符合特定格式，因而不会出现在文档相应位置。

<!-- image-->

下面介绍如何写能被Doxygen 识别的注释以及一些基本用法。

## 基本格式

在c++中Doxygen 会识别以下三种注释格式：  
Javadoc-style

```css
1 /** 1
2 * This is a Javadoc-style comment
3 */
```

Qt-style

```javascript
1 /*!
2 2* This is a Qt-style comment
3 */
```

Extended c++ style

1 /// This is the so-called extended c++-style comment   
2 //! This also works!

以上三种格式本身均是合法的C++ 注释，只是在和//的基础上添加了额外的'\*’，’或"’。一旦文档中出现了以上三者之一，Doxygen 便会将之后的注释内容填入文档的相应位置。这三种基本格式是所有 Doxygen用法的基础，所有的进阶功能（如group，module，equation 等）均需在这三者之一开启的注释行/块内实现。这三种基本格式可以混用，开发者可以按自己的习惯选择一种或多种使用。

Doxygen 默认注释作用于之后的类、变量或函数（允许有空行)。如我们希望注释作用于之前的对象，则可在基本格式后立刻添加'<'符号，例如

1 //! this comment line applies to print()，even though there's an empty line below   
2   
3 void print();   
4 ///< this comment line also applies to print(),not get(); note the '<' character!   
5 int get();

## 详细与简略描述

如之前例子所示，Doxygen 自动生成类的文档时会先把成员函数与变量罗列在开头，随后才是每个成员的介绍（当然，由于之前的例子里没有采用doxygen 格式，详细描述为空)。只要使用了基本格式，Doxygen 就会自动将注释附在对应成员处。比如若将之前构造函数前的//注释替换为/：

1 /// Initializes a Circle object with a given radius.   
2 2 /// A unit circle is assumed by default.   
3 Circle(double r = 1.0);

这些内容即会出现在原先为空的Constructor &Destructor Documentation 下 CircleO的描述里:

## Circle Class Reference

Public Member Functions | Static Public Member Functions |   
Private Atributes | Static Private Attributes |List of all members

<!-- image-->

Circle (double r=1.0)   
Initializes a Circle object with a given radius. More...

Static Public Member Functions

static void rescale (Circle \*circ, double factor)   
Rescales a Circle object. More...

Private Attributes

doubler_   
radius of the Circle object More..

static constexpr double Pl = 3.14159265358979324

A class that represents a circle.   
This toy class willbe used to demonstrate the usage of Doxygen

Constructor & Destructor Documentation

• Circle()   
Circle::Circle (double r = 1. 0)   
Initializes a Circle object with a given radius.   
A unit circle is assumed by default.

Doxygen 还允许让一些简略描述出现在类文档的开头部分。有两种方式可以实现这一目的。较为正式的方式是在注释块里添加 @brief 或者\brief。以这种方式指定的简略描述可以跨多行（如下面开头Circle类的注释)。另一种较为轻便的方式是使用单行的 extended C++ style 注释，如下面Circle的构造函数。【多行的extended c++style 注释只会被当做详细描述】

1 /\*\*   
2 \* @brief A class that represents   
3 \* a circle.   
4 \*   
5 \* This toy class will be used to demonstrate the usage of Doxygen   
6 \*/   
class Circle {   
8 public:   
9 /// Initializes a Circle object with a given radius.

10 /\*!   
11 \* A unit circle is assumed by default.   
12 \*/   
13 Circle(double r = 1.0);   
14   
15 /\*!   
16 \* \brief Rescales a Circle object.   
17 \*   
18 \* The radius of the given Circle object is multiplied by the given factor.   
19 \*   
20 \*/   
21 static void rescale(Circle\* circ,   
22 double factor   
23 ）；   
24   
25 private:   
26 double r_; ///< radius of the Circle object   
27 static constexpr double PI = 3.14159265358979324;   
28 }；

以上代码生成的Doxygen文档在开头罗列部分会出现对应的简略描述；详细介绍部分则会包括所有的注释（包括brief)。

·rescale()   
static void Circle:rescale ( Circle \* circ,   
double factor   
static   
Rescales a Circle object.   
Theradius of the given Circle object is multiplied by the given factor.   
Parameters   
[in,out] circPointer to the Circle object to be rescaled.   
[in] factor Radius rescaling factor.

## 函数参数

一个开发者写的函数经常会被很多其他开发者使用。除对函数整体功能的描述外，每个参数含义的注释也十分重要。Doxygen支持两种给函数参数添加注释的方式。较为正式的方式是在注释块内添加@param 或\param:

```c
1 /*!
2 * \brief Rescales a Circle object.
3*
4 * The radius of the given Circle object is multiplied by the given factor.
5 *
6* @param [in,out] circ Pointer to the Circle object to be rescaled.
7* @param [in] factor Radius rescaling factor.
8*/
9 static void rescale(Circle* circ,
```

10 double factor  
11 ）；

较为轻便的方式是在函数声明的参数列表里直接添加（加在参数后注意要使用'<’）

1 /\*!   
2 \* \brief Rescales a Circle object.   
3\*   
4 \* The radius of the given Circle object is multiplied by the given factor.   
5\*/   
6 static void rescale(Circle\* circ,///<[in,out] Pointer to the Circle object to be rescaled.   
7 double factor //<[in] Radius rescaling factor.   
8 ）；

[in],[out]或[in,out]可以提示该参数是输入还是输出参数。这并非 Doxygen 的强制要求，但十分推荐。以上两种参数注释方式对Doxygen而言是等价的，开发者可按习惯选择一种使用，生成的函数描述如下：

+ PI   
constexpr double Circle::Pl = 3.14159265358979324   
A constant that represents the value of π.

## 数学公式

有时一些函数严格对应了某个公式，这种情况下直接用数学公式作为注释的效果可能胜过大段的文字描述。Doxygen 允许开发者以Latex 的语法实现公式注释。如有在文字中嵌人公式的需求，可以采用\f\$...f\$:

1 /// A constant that represents the value of \f\\$ \pi \f\\$.   
2 static constexpr double PI = 3.14159265358979324;

·area0   
double Circle::area （）const   
Returns the area of the current Circle object.   
The area of the a circle is given by   
S=πr²

单行公式由f[.f]实现:

1 /\*!   
2 \* @brief Returns the area of the current Circle object.   
3 \*   
4 \* The area of the a circle is given by   
5 \*   
6 \* \f[S = \pi r^2 \f]

7 \*/   
8 double area() const;

• circumference()   
double Circle:circumference （） const   
Returns the circumference of the current Circle object.   
The circumference of thea circle is given by   
L=πd   
=2πr

如有多行公式的需求，Doxygen 支持调用eqnarray 等latex 数学环境：

1 /\*!   
2 \* @brief Returns the circumference of the current Circle object.   
3 \*   
4 \* The circumference of the a circle is given by   
5 \*   
6 \* \f{eqnarray\*}{   
7 \* L &= \pi d \   
8 \* &= 2 \pi r   
9 \*\f}   
10 \*   
11 \*/   
12 double circumference() const;

A class that represents a circle.   
This toy class will be used to demonstrate the usage of Doxygen   
List of functionalities:   
1. initialize a Circle object   
。 with a given radius   
○a unit circle is assumed by default   
2. get the radius of the current Circle object   
3. get the area of the current Circle object   
4. get the circumference of the current Circle object   
5. rescale a Circle object by a factor

【在配置文件中采用“USE_MATHJAX=YES”可以调用mathjax 渲染公式得到更好的视觉效果】

## 列表

Doxygen还有大量用法可供用户生成理想的文档。

对齐的'-,'+’或'\*’可自动生成列表(bullet list)；'-#则可用来生成带序号的列表:

1 /\*\*   
2 \* @brief A class that represents   
3 \* a circle.   
4 \*   
5 \* This toy class will be used to demonstrate the usage of Doxygen   
6 \*   
7 \* List of functionalities:   
8 \* -# initialize a Circle object with a given radius   
9 \* - a unit circle is assumed by default   
10 \* -# get the radius of the current Circle object   
11 \* -# get the area of the current Circle object   
12 \* -# get the circumference of the current Circle object   
13 \* -# rescale a Circle object by a factor   
14 \*   
15 \*/   
16 class Circle \*/

Public Member Functions   
Circle (double r=1.0)   
Initializes a Circle object with a given radius. More...   
Properties   
Functions that return the properties of the current Circle object.   
double radius 0 const   
Gets the radius of the current Circle object. More...   
double area ( const   
Returns the area of the current Circle object. More..   
double circumference () const   
Returns the circumference of the current Circle object. More...

## Member Group & Module

变量或函数可以由@...“@ 归类为 member group 并附上相应注释，这能使文档更有层次感：

```cpp
1 //@{ @name Properties
2 /// Functions that return the properties of the current Circle object.
3
4 /// Gets the radius of the current Circle object.
5 double radius() const { return r_; }
6
7 /*!
8* @brief Returns the area of the current Circle object.
9*
10 * The area of the a circle is given by
11 *
12 *\f[S = \pi r^2 \f]
13 */
14 double area() const;
15
```

```c
16 /*!
17 * @brief Returns the circumference of the current Circle object.
18 *
19 * The circumference of the a circle is given by
20 *
21 * \f{eqnarray*}{
22 * L &= \pi d \
23 * &= 2 \pi r
24 *\f}
25 *
26 */
27 double circumference(） const;
28 ///@}
```

## Shape Module

## 有共性的类之间亦可归类组成Modules:

```scss
1 /// @defgroup shape Shape Module
2 /// @brief A module that contains various shape classes.
3 ///
4 /// This module contains the following classes:
5 /// - Circle
6 /// - Rectangle
7 /// - Triangle
8
9 /// @addtogroup shape
10 /// @{
11 class Circle {
12 ：
13 }；
14 ///@}
15
```

16 /// @addtogroup shape   
17 /// @{   
18 class Rectangle {   
19 ：   
20 }；   
21 ///0}   
22   
23 /// @addtogroup shape   
24 /// @{   
25 class Triangle {   
26 ：   
27 }；   
28 ///@}

<!-- image-->

更多用法可参考官方文档Doxygen Manual:Overview

## 如何生成文档

上一节介绍了Doxygen 支持的注释格式以及一些常见的使用方式。以下以Ubuntu用户为例介绍让开发者在本地生成 html 文档的方法:

1.安装 doxygen 与 graphviz

## 2.生成Doxygen 的配置文件

1 doxygen -g <your_preferred_doxyfile_name>

默认的配置文件名为Doxyfile。

## 3.修改配置文件

配置文件内已经包含介绍各选项的注释。常用的选项包括：

INPUT：用以生成文档的文件夹或文件路径

RECURSIVE：是否遍历给定路径下的所有子文件夹

EXCLUDE：是否要排除一些子文件夹或子文件

EXTRACT_PRIVATE,EXTRACT_STATIC,EXTRACT_PRIV_VIRTUAL:文档是否要包括私有成员/静态成员/私有虚函数

CALL_GRAPH,CALLER_GRAPH:文档是否要包含函数调用/被调图

PREDEFINED：预编译宏

MACRO_EXPANSION：是否要展开宏定义

EXPAND_ONLY_PREDEF：是否只展开PREDEFINED 的宏定义

一个可供参考的Doxyfile可见于Doxygen 使用 (赵天琦)

完整的配置选项可参考官网DoxygenManual:Configuration

4.运行 doxygen

假设配置文件叫my_doxyfile:

1 doxygen my_doxyfile

## 7.3.3 附录：Doxygen 与VS Code

Visual Studio Code 上有 doxygen 插件 Doxygen Documentation Generator - Visual Studio Marketplace 可供 VSCode 用户方便使用。

【注：前文中有提到对doxygen 而言三种基本注释格式可以混用；混用的注释会合并在一起作用于注释后的对象。这个特性与VS Code中鼠标悬停于类或函数上时出现的预览窗口不兼容；如注释格式混用则会导致前一种格式的注释被忽略】

## 7.4ABACUS线上文档输人参数撰写规范

## 7.4.1 背景

ABACUS 代码仓库地址：https://github.com/deepmodeling/abacus-develop

ABACUS 线上文档地址：https://bacus.deepmodeling.com/en/latest/

(ABACUS线上文档是包括开发者在内的所有用户了解 ABACUS软件使用方法的重要渠道。）

本文档关注：“关键字”部分的文档

对应的文档地址：https://abacus.deepmodeling.com/en/latest/advanced/input_files/input-main.html

对应的代码地址：htps:/github.com/deepmodeling/abacus-develop/blob/develop/docs/advanced/input_files/inputmain.md

更新方式：目前对该文档的更新同样是通过PR的方式提交到ABACUS 的Github仓库上，因此任何人都可以通过提交PR的方式对文档做出修改，我们也鼓励用户如果发现问题后可以主动修复文档。

## 7.4.2 规范格式

每个输入参数都应遵循以下格式：

Name of Parameter Type: Availability: (可选) Description: Default: Unit: (可选)Example: (可选)

按照上述顺序排列，上述所有字段需加粗，后面使用英文冒号（不加粗)，冒号后空一格开始写内容（不加粗)

上述关键字用黑色实心圆，关键字后的介绍不用加粗

错误示范：Type:Integer 错误示范：Type: integer 错误示范：Type: integer 正确示范：Type:Integer 错误示范：Default:O 错误示范:Default:false 正确示范：Default:False

## Name of Parameter

这个参数代表了参数的名称，建议起参数名称的时候遵从以下规范（xxx由设定参数名的开发者指定)：

·输出某物理量的参数：建议以out_xx为参数名，其中out是output 的缩写。可参考的变量名有out_chg（输出电荷密度)，out_pot（输出势函数)，out_dm（输出密度矩阵density matrix)

·收敛阈值相关参数：建议以 xxx_thr 为参数名，其中thr是threshold的缩写。可参考的变量名有 scf_thr（电子自洽迭代的收敛阈值)，force_thr（结构驰豫时原子中的最大受力域值)，和 stress_thr（晶格优化时的应力收敛阈值)

·计算某物理量的参数：建议以cal_xxx 为参数名，其中cal是calculate 的缩写

·初始化某物理量的参数：建议以init_xxx为参数名,其中init是initialize的缩写。可参考的变量名有init_wfc(初始化波函数)，init_chg(初始化电荷密度)，init_vel(初始化原子速度)

·结构驰豫相关参数：建议以relax_xxx为参数名，其中relax 代表（结构）驰豫

·并行策略相关参数：建议以 xxxpar 为参数名，其中 par是代表 parallel，即并行。可参考的变量名有 kpar（k点并行)，bndpar（能带并行)

·其它参数：建议先寻找ABACUS目前支持的参数名来设计相关参数

## Type

Type:冒号之后首字母大写,输人参数的种类,有Integer,Real和 String三种类型。对于Boolean类型,Default字段后的内容统一写：True 或False（首字母大写)，且先写True，再写False

注意Real类型不要写成double 或者float类型

## Availability (Optional)

如果是完整的一句话就首字母大写，如果是单词或短语就小写

## Description

。 冒号之后首字母大写

·如果有 Note 的标注，可在Description 字段下，单独写一个Note，不用加粗。

Description 字段里如果有涉及到1和O是表示 True 或False，统一改成 True或 False。

·字段如果是完整句子，末尾要加英文句号":”。若是单词和短语则不加。其他字段均不加。

。 String类型的参数直接列出所有选项

·当一个参数有多种取值时，按照从小到大顺序描述不同的取值，例如 propagator这个参数，就在description 里面描述该参数是0、1、2三种情况下的波函数演化方法。

·不写“when set to”、“if set to”，直接写参数后，再写该参数的描述

## Default

参数默认的数值，分成有单位的和没有单位的两种。如果是小数，建议采用科学计数法表示，“e”一律小写，如:2e-6

## Unit (Optional)

如有需要使用Unit来标定输入参数的单位，紧接在Unit 后面描述单位，另外内容里如果涉及到写单位，需要用空格隔开，如：U(eV)

单位撰写需符合标准：

错误示范：KBar 错误示范：ev正确示范：kbar正确示范：eV

## Example (Optional)

如有案例，可添加相关例子的链接和简单描述。如果需要，可以添加上述的Example字段描述案例。

## 7.4.3 其他问题

·所有简写应改大写，以及后面用括号写出全称，每一个关键字的描述出现了简写均需要全称，如：“soc”改成“SOC（spin-orbital-coupling)”。即使最常见的如 PW、MD 等也均写出全称

·对参数的描述保持客观，如：不写“An important parameter”这种话

·相关的参数建议放在文档上靠近的地方。举例：symmetry 参数是控制是否打开对称性操作的参数，而 symme-try_prec 是用来控制对称性操作的精度，这两个参数就可以放在一起先后介绍，并且在介绍的时候，可以提及相关的参数

。 一些太长的输出，单独开辟一个专区去介绍

·关键字告诉了用户输出了文件，建议告诉用户文件名称和文件保存的路径

。 注意语法问题

。 标点符号格式要一致

。 注意不要出现乱码

以上若有问题或者建议，欢迎写信给作者！虽然我们邮箱基本不会收到类似邮件，但开源精神的意义就在于提供了这种可能性，然后坚信更好的事情会到来。

## 7.5ABACUS代码存放规范

## 7.5.1 背景

截止到 ABACUS 3.4 版本，ABACUS 的 Github 仓库（htps:/github.com/deepmodeling/abacus-develop）代码已超过30万行，涉及上千个源代码文件，有时一周内在Github仓库上就会 Merge近20个PR，涉及数千乃至上万行代码的修改。随着社区规模不断壮大，为了更好的在各个开发者之间协同合作，在过去已有的一些代码管理的共识的基础上，我们提出这个代码存放规范文档。

## 7.5.2代码目录

进入主仓库之后，ABACUS 的源代码目录位于 source 文件夹下：

\\$ tree source -d -L 1  
2 source  
3 module_base # 基础模块  
4 module_basis #LCAO 和 PW 基组模块  
5 module_cell #晶胞模块  
6 module_elecstate # 电子波函数相关性质模块  
7 module_esolver #能量求解器模块  
8 module_hamilt_general # 对于 PW 和 LCAO 基组共享的哈密顿量模块  
9 module_hamilt_lcao #LCAO 基组下哈密顿量的相关模块  
10 module_hamilt_pw #PW 基组下哈密顿量求解的相关模块  
11 module_hsolver # 对角化求解矩阵特征值/特征向量等哈密顿求解器模块  
12 module_io # 输入输出模块  
13 module_md # 分子动力学模块  
14 module_psi # 波函数模块  
15 module_relax # 结构优化模块  
16 module_ri #Resolution of identity 相关的 Beyond DFT 的模块

其中，14个主模块的核心功能如上面注释所言，代码存放建议按照模块化管理，相关功能的代码存放到对

应的模块当中。如果需要涉及到新建模块以及更新模块位置等需求，应该按照 ABACUS 的Github 仓库 Issues 处理流程·GitBook 中相关规范要求去Github 仓库提交ISSUE 并邀请开发者讨论来进行。

## 模块内部代码存放规范

以 module_base 目录为例，该目录下包含了ABACUS 中内存管理，错误处理，基础通讯以及基础算法等等功能的代码实现，是ABACUS 中最底层的代码，其他模块大多依赖于 module_base。该模块中的代码分布如下，相应文件夹功能如注释所言：

```shell
\$ tree source/module_base -d -L 1
2 source/module_base
3 CMakeLists.txt
4 abfs-vector3_order.cpp #基础代码
5 abfs-vector3_order.h #基础代码
6 ：
CMakeLists.txt # 负责 module_base 中 cmake 编译的 CMakeLists.txt
8 kernels #module_base 中的异构计算代码
9 libm #libm 模块，包含一些优化过后的数学库运算
10 module_container #ABACUS 中底层容器 Tensor 的实现
11 module_mixing #charge-mixing 实现
12 test #module_base 中的单元测试
13 test_parallel #module_base 中并行相关的单元测试
14
15 ylm.cpp #基础代码
16 ylm .h #基础代码
```

我们提出如下的模块代码存放规范：

1.基础代码存放规范：模块中的基础代码（除去部分特殊代码外的大部分代码）应直接存放到当前模块目录下，头文件以.h命名；源文件以.cpp 命名；代码规范应该遵循 ABACUS 开源项目C++ 代码规范·GitBook 中的要求。

2.异构计算代码存放规范：模块中涉及到异构计算（CPU结合GPU计算）部分的代码应直接存放于当前模块的kernels目录下；这部分代码的结构将会在稍后详细介绍。

## 3.子模块代码存放规范：

在主模块下同样可能存在一些子模块，它们通常是具有特定功能的代码的集合：以module_base 举例，libm,module_container以及module_mixing 分别表示了libm 模块（后续建议改成 module_libm 模块），ABACUS 中底层 Tensor容器模块（用来方便在不同硬件上存放数据）以及charge-mixing 相关的功能模块，它们为 module_base模块下的子模块，子模块中同样也可能有类似于主模块的文件代码排布。

如果要在当前模块中新建或者调整子模块应当去Github 仓库提交相关的ISSUE 讨论。

## 4.单元测试代码存放规范：

模块中代码的单元测试应当存放到当前模块的test目录下，特殊情况可以存放到以test开头的目录下面。例如，test_paralel存放了一些并行计算的单元测试，因为test 目录下面的CMakeList 文件有时候会把MPI宏定义去除，再比如test_mpi里面只测试了mpi代码，没有打开OPENMP。module_base下面还有一个针对数值精度的测试，命名成了test_precision。

当前模块的子模块的单元测试应当存放到相应子模块的test目录下;

5.CMakeLists.txt存放规范：各个模块以及子模块通过各自目录中的CMakeLists.txt 管理CMake 编译结构，添加或修改代码要注意维护CMake编译的完备性，此外还应该维护Makefile 编译的实现，确保代码可以通过Github 的 CI测试。

## 异构计算代码存放规范

当前 ABACUS 中PW部分代码已经实现了初步的GPU化，Source 目录中包含GPU的module 如下，相应文件夹功能如注释所言：

module_base # 基础模块  
kernels #module_base 模块下的异构计算代码  
3 cuda #module_base/kernels 中函数的 CUDA 实现  
4 rocm #module_base/kernels 中函数的 ROCm/DCU 实现  
5 test #module_base/kernels 中函数的单元测试  
6 libm # 子模块 libm  
test #libm 的单元测试  
8 module_container # 子模块，负责实现 ABACUS 中的底层 Tensor 容器  
9 ATen # 定义了 Tensor 容器的基本实现  
10 base # 定义了一些 Tensor 对象无关的基础函数  
11 test #module_container 的单元测试  
12 test #module_base 的单元测试  
13 data #module_base 的但愿测试依赖的数据  
14 test_parallel #module_base 的并行测试  
15 <strong>...</strong>  
16 module_elecstate # 电子波函数相关性质模块  
17 kernels #module_elecstate 中相关的异构计算代码  
18 cuda  
19 rocm  
20 test  
21 module_charge #charge 子模块  
22 module_dm #dm 子模块  
23 test #module_dm 的单元测试  
24 potentials #potentials 子模块  
25 test #module_elecstate 中的单元测试  
26 support # 放置单元测试需要的输入文件  
27 test_mpi # 只测试 mpi 并行代码的单元测试  
28 <strong>...</strong>  
29 module_hamilt_pw #PW 基组下哈密顿量求解的相关模块  
30 hamilt_ofdft #OFDFT 子模块  
31 hamilt_pwdft #PWDFT 子模块  
32 kernels  
33 operator_pw  
34 test  
35 hamilt_stodft  
36 module_hsolver # 对角化求解矩阵特征值/特征向量的模块  
37 genelpa  
38 kernels  
39 cuda  
40 rocm  
41 test  
42 test  
43 <strong>...</strong>  
44 module_psi # 波函数存储模块

为了更进一步的规范异构计算代码在ABACUS 中的维护和管理，以module_base/kernels 举例，针对异构计算代码的位置和目录结构做出如下规范：

```ruby
module_base/kernels/
2 cuda # 存放算子的 CUDA 实现
3 math_op.cu # 算子的 CUDA 实现存放于 cuda 目录下
4 math_op.cpp ＃ 算子的 CPU 实现
5 math_op.h # 算子的声明
6 rocm # 存放算子的 ROCM 实现
7 math_op.hip.cu # 算子的 ROCM 实现存放于 rocm 目录下
8 test # 提供了算子的单元测试
9 CMakeLists.txt
10 math_op_test.cpp
```

1.异构计算代码存放规范：在每个module下，异构计算相关的代码都应该存放在kernels目录下。异构计算代码都是以算子（OP）的形式存在的，一个算子包含了：CPU，CUDA以及ROCM的实现，使得该算子可以作为统一的计算接口供ABACUS 调用。

2.算子的声明和CPU实现存放规范：以 module_base 目录举例：module_base/kernels 目录应该直接存放当前 module 中涉及到的异构计算算子的声明.h文件和其CPU 实现.cpp 文件;

3.算子的CUDA 实现存放规范：kermels 目录下的cuda子目录中相应的.cu 文件存放算子的CUDA 实现;

4.算子的 ROCM实现存放规范：kernels 目录下的 rocm子目录存放算子的 ROCM实现;

5.算子的单元测试存放规范：kermels 子目录下的test目录存放当前 kernels 中涉及到的全部算子的单元测试。

## 7.5.3总结

我们提倡在ABACUS 中将各个模块的代码按照上述规范建议进行存放，目录结构可以延续上述风格以便代码识别；涉及到代码目录的问题欢迎到Github仓库进行讨论交流。

## 7.6 如何在ABACUS中新增一个输人参数（截至v3.5.3)

## 7.6.1 在 input 类中申明和定义

## 声明

路径：source/module_io/input.h

首先现在input.h中添加相关输入参数的定义，如nelec_delta:

1 //======:   
2 // electrons / spin   
3 //==

4 std::string dft_functional; // input DFT functional.   
5 double xc_temperature; // only relevant if finite temperature functional is used   
6 int nspin; // LDA ; LSDA ; non-linear spin   
double nupdown = 0.0;   
8 double nelec; // total number of electrons   
9 double nelec_delta; // change in the total number of electrons   
10 int lmaxmax;

## 设置初始值

路径：source/module_io/input.cpp

接下来你需要在 Input::Default(void)和 Input:Default_2(void)函数里对参数进行初始化，其中:

Input::Default(void)——定义单个变量的初始值

Input:Default_2(void)——根据部分参数的值，对一些有关联的参数重新赋值，如nelec_delta:

//-   
2 // electrons / spin   
3 //----   
4 dft_functional = "default";   
5 xc_temperature = 0.0;   
6 nspin = 1;   
nelec = 0.0;   
8 nelec_delta = 0.0;   
9 lmaxmax = 2;

## 读取

路径：source/module_io/input.cpp

在Input::ReadO中定义从 INPUT 文件中读取相应的参数，例如：

1 else if (strcmp("nelec_delta", word) == 0)   
2 {   
3 read_value(ifs, nelec_delta);   
4 }

## MPI广播

路径：source/module_io/input.cpp

读取完之后，需要定义如何在Input:Bcast(将读取的参数广播到所有的节点，原因是ABACUS 程序从一个进程读入参数，之后通过MPI的Bcast函数广播到其它的进程。例如：

Parallel_Common::bcast_double(nelec_delta);

注意这里你需要根据参数的类型，使用不同的语句：

Parallel_Common: :bcast_bool(out_wannier_wvfn_formatted);   
2 Parallel_Common: :bcast_string(dft_functional);   
3 Parallel_Common: :bcast_double(xc_temperature);

4 Parallel_Common::bcast_int(nspin);

## 输出

路径：source/module_io/write_input.cpp

ABACUS 会在 Input:Print(const std::string &fn)中做参数的输出，你也需要在这个函数中进行添加:

```rust
1 ModuleBase::GlobalFunc:0UTP(ofs, "nelec_delta", nelec_delta,"change in number of total electrons");
```

## 添加测试

路径：

TEST_F(InputParaTest,Bcast)既测试了初始值的设定，也同样测试了Bcast 的有效性，你也要将新增参数加入：

```sql
1 EXPECT_DOUBLE_EQ(INPUT.nelec_delta,0.0);
```

这里同样需要考虑不同的数据类型：

1 EXPECT_TRUE(INPUT.out_Wannier_mmn);   
2 EXPECT_FALSE(INPUT.out_Wannier_unk) ;   
3 EXPECT_DOUBLE_EQ(INPUT.neleC，0.0)；   
4 EXPECT_EQ(INPUT.basis_type， "pw");

参数输出的测试

TEST_F(write_input,General1)测试了ABACUS 参数输出的正确性，例如：

1 EXPECT_THAT(output， testing: :HasSubstr("nelec   
2 0 #input number of electrons"));

这里一定要注意空格和对齐。

## 7.6.2将input成员转换为全局变量

参数参与计算一般会先转换为全局变量GlobalV，再通过传参赋值给负责计算的类（但是注意将来我们会逐步弃用GlobalV，所以这部分将来可能会变化）。

声明

路径：source/module_base/global_variable.h

在GlobalV这个命名空间中申明新增的参数，例如：

1 extern double nelec_delta;

## 设置初始值

1 double nelec_delta = 0;

## 利用Input类的成员为GlobalV中成员赋值

在Input_Conv:Convert(void) 中为Global 的参数赋值，例如：

```rust
1 GlobalV::nelec = INPUT.nelec;
```

注意：Input类一个全局类，只要调用了头文件即可使用。

## 添加测试

```sql
1 EXPECT_EQ(GlobalV::nelec_delta,0);
```

## 7.6.3 添加文档

路径：docs/advanced/input_files/input-main.md

每个新参数的PR 必须包含相应的文档，否则不会被接收。请在 input-main.md 中添加参数描述。

## 7.7如何在ABACUS中新增一个输人参数（v3.7.0后）

## 7.7.1开发者须知

声明参数并给出初始值

```c
1 bool init_vel = false; ///< read velocity from STRU or not liuyu 2021-07-14
2 double symmetry_prec = 1.0e-6; ///< Liuxh add 2021-08-12, accuracy for symmetry
3 bool symmetry_autoclose = true; ///< whether to close symmetry automatically
4 ///< when error occurs in symmetry analysis
5 double nelec = 0.0; ///< total number of electrons
6 double nelec_delta = 0.0; ///< change in the number of total electrons
```

在声明的同时给出默认值，如nelec 默认为0.0。

注：对于任何类成员变量的double,int,bool等一般类型变量，在定义时就建议给一个初始值。如果类成员没有初始值，很可能会有难以察觉的bug

注：string 类型变量的初始值不能为"”,如果想自动设置，可以设置初始值为"auto”,"none”等。

## 在参数列表中添加参数

路径module_io/read_input_item_\*.cpp（不同的参数分好了类，其对应不同文件，需要加到属于的文件里面),如 nelec 就在 read_input_item_general.cpp 中

添加nelec参数的item

1 Input_Item item("nelec");

之后依次添加： (\*为必填)

注释（annotation\*)：该注释会被打印在OUT.\*文件夹的 INPUT中

```typescript
1 item.annotation = "input number of electrons";
```

重新赋值函数（reset_value）：如果其他某些参数满足特定条件是，需要对当前参数（nelec）进行修改的函数，该函数一定会执行。书写规范要求：只把修改当前参数的函数部分定义在当前参数的reset_value，如果要修改其他参数，请定义在其他参数的 reset_value。

```javascript
1 item.reset_value = [](const Input_Item& item, Parameter& para) {
2 if(para.input.somecondition)
3 {
4 para.input.nelec = 0.0;
5 }
6 }；
```

检验函数（check_value）：用于检查目前参数是否合适，如果不合适就报错退出（warning_quit)，该函数一定会执行。

```cpp
1 item.check_value = [](const Input_Item& item, const Parameter& para) {
2 if(para.input.nelec < 0)
3 {
4 ModuleBase::WARNING_QUIT("ReadInput", "nelec should be no less than 0.0");
5 }
```

获取最终打印值函数（get_final_value\*）：用于给 stringstream final_value 赋值，最终 final_value.strO会被打印在OUT.\*的 INPUT文件中，该函数一定会执行。

1 item.get_final_value = [](Input_Item& item, const Parameter& para) {  
2 item.final_value << para.input.nelec;  
3 }；

以及如果是并行版，需要添加广播函数（bcastfuncs\*）：如何调用 Parallel_Common 进行 bcast 的函数，该函数一定会执行。该函数不一定只传输该参数，还可以传输其他引入的非 INPUT参数，例如gamma_only_local这个变量不是 INPUT参数，但是其由 gamma_only 和 basis_type 共同决定取值，这种额外引入的参数也需要在gamma_only 或 basis_type 处添加他的 bcast 函数。

1 #ifdef __MPI   
bcastfuncs.push_back([] (Parameter& para) {   
3 Parallel_Common: :bcast_bool(para.input.nelec);

}）   
5 #endif

最后将item 添加到参数列表中：

1 this->add_item(item);

总结：

```cpp
1{
2 Input_Item item("nelec");
3 item.annotation = "input number of electrons";
4 item.read_value = [](const Input_Item& item, Parameter& para) {
5 para.input.nelec = std::stoi(item.str_values[0]);
6 }；
7 item.reset_value = [](const Input_Item& item, Parameter& para) {
8 if(para.input.somecondition)
9 {
10 para.input.nelec = 0.0;
11 }
12 }；
13 item.check_value = [](const Input_Item& item， const Parameter& para） {
14 if(para.input.nelec < 0)
15 {
16 ModuleBase::WARNING_QUIT("ReadInput", "nelec should be no less than 0.0");
17 }
18 }；
19 item.get_final_value = [](Input_Item& item, const Parameter& para) {
20 item.final_value << para.input.nelec;
21 }；
22 #ifdef __MPI
23 bcastfuncs.push_back([] (Parameter& para) {
24 Parallel_Common: :bcast_bool(para.input.nelec);
25 })；
26 #endif
27 this->add_item(item);
28 }
```

以上是完整的参数添加实例，可以根据需要自己修改函数，然而实际上大部分参数的 bcastfuncs、get_final_value、read_value 函数填写是几乎一致的，他们有共同的形式，为了书写方便，可以利用 mod-ule_io/read_input_tool.h定义的宏函数来进行书写的简化：

1.对于形如

```javascript
1 bcastfuncs.push_back([] (Parameter& para) {
2 Parallel_Common:bcast_bool(para.input.nelec);
3 })；
```

的bcastfuncs 函数书写可以分别使用add_double_bcast，add_int_bcast，add_bool_bcast，add_string_bcast,

---

## 第 341–360 页

add_doublevec_bcast,add_intvec_bcast,add_stringvec_bcast 函数进行代替

## 2.对于形如

1 item.get_final_value = [](Input_Item& item, const Parameter& para) {   
2 item.final_value << para.input.nelec;   
3 }；   
4 bcastfuncs.push_back([] (Parameter& para) {   
5 Parallel_Common::bcast_bool(para.input .PARAMETER);   
6 })；

的 get_final_value 与 bcastfuncs 函数的合并书写可以使用 sync_double，sync_int，sync_bool，sync_string,sync_doublevec,sync_intvec,sync_stringvec 函数进行代替

## 3.对于形如

```cpp
1 item.read_value = [](const Input_Item& item, Parameter& para) {
para.input.nelec = std::stoi(item.str_values[0]);
}；
4 item.get_final_value = [](Input_Item& item, const Parameter& para) {
5 item.final_value << para.input.nelec;
6 }；
bcastfuncs.push_back([](Parameter& para) {
8 Parallel_Common: :bcast_bool(para.input .nelec);
9 })；
```

的read_value,get_final_value 与 bcastfuncs 函数的合并书写可以使用read_sync_double,read_sync_string,read_sync_int,read_sync_bool 函数进行代替

因此添加nelec 的代码可以简化成：

1 {   
2 Input_Item item("nelec");   
3 item.annotation = "input number of electrons";   
4 read_sync_double(nelec);   
5 item.reset_value = [](const Input_Item& item, Parameter& para) {   
6 if(para.input.somecondition)   
7 {   
8 para.input.nelec = 0.0;   
9 }   
10 }；   
11 item.check_value = [](const Input_Item& item， const Parameter& para) {   
12 if(para.input.nelec < 0)   
13 {   
14 ModuleBase::WARNING_QUIT("ReadInput"， "nelec should be no less than 0.0");   
15 }   
16 }；   
17 this->add_item(item);

## 参数的使用

目前已经舍弃使用INPUT与GloabIV::，请勿在里面添加新变量！

需要用到参数的地方，请以只读的方式使用

1 int nbands = PARAM.inp.nbands； //只读的访问元素

而其不能作为修改的对象，这时编译不能通过：

1 PARAM.inp.nbands = O； //会报错

## 添加测试

读入测试

<u>source/module_io/test/support/INPUT</u>

在INPUT中添加这一参数：

1 nelec

10 #input number of electrons

在 TEST_F(InputParaTest,ParaRead) 测试这一读入是否正确，例如：

1 EXPECT_EQ(PARAM.inp.nelec， 10);

特殊场景的 reset_value 和 check_value 测试

路径：source/module_io/test/read_input_item_test.cpp

在 TEST_F(InputTest,Item_test)添加特定参数的 reset_value 函数和 check_value 的覆盖性测试。

## 添加文档

路径：docs/advanced/input_files/input-main.md

每个新参数的PR 必须包含相应的文档，否则不会被接收。请在input-main.md 中添加参数描述。

## 7.7.2 设计文档参考

参考：林霈泽博士提的方案github.com

## 背景

v3.7.0之前添加参考如何在 ABACUS 中新增一个输入参数（截至v3.5.3)

原 input.cpp

当添加一个新的 INPUT参数，我们需在 input类里做以下事情：

A.首先在Default函数中给初始值;

B.再在 read 函数中的if中加一个判断分支：

1 if (strcmp("suffix", word) == O) // out dir   
2{   
3 read_value(ifs, suffix);   
4

C.再在Bcast 函数中添加 bcast

D.（大部分）在GlobalV定义一个相同功能的变量，并给个初始值（实际上这个初始值没有任何用，还容易让人误解到底这个和input.cpp 哪个是初始值)

E.再在 input_conv.cpp 中转化成 GlobalV 的变量

F.再在write_input.cpp 添加一行代码使其可以输出到 OUT.ABACUS 文件夹中 INPUT

## 痛点:

每加一个参数，对于ABACUS 主代码需“翻山越岭”地添加代码，不易管理

input.cpp 的变量和GlobalV 的变量大部分重复，没有必要，且容易误解

大量的if分支改进思路：

将相同的参数代码集中起来

input类只实现读的功能，不存储参数，参数由另一个类存储

GlobalV

其实代码有全局变量不可怕，可怕的是全局变量会在运行中改变，而部分GlobalV就是这样，因此需要限制他人在初始化参数之后改变的行为。

目前很多也使用了public 的类静态成员变量，这其实和全局变量是一样的。

## 重构设计

## Parameter 类

该类只存储 ABACUS 运行的参数，将代替GlobaIV 的功能，该类将成员设成私有变量，只给 ReadInput 类修改的权限，其他类只有访值权限，没有修改权限：

```cpp
1 class Parameter
2{
3 public:
4 Parameter(){};
5 ~Parameter(){};
6 public:
7 // We can only read the value of input， but cannot modify it.
8 const Input_para& inp = input;
9 // We can only read the value of mdp,but cannot modify it.
10 const MD_para& mdp = input.mdp;
11 // We can only read the value of other parameters, but cannot modify it.
12 const System_para& globalv = system;
13
14 // Set the rank & nproc
15 void set_rank_nproc(const int& myrank, const int& nproc);
16 // Set the start time
17 void set_start_time(const std::time_t& start_time);
18
19 private:
20 // Only ReadInput can modify the value of Parameter.
21 friend class ModuleIO::ReadInput;
22 // INPUT parameters
23 Input_para input;
24 // System parameters
25 System_para system;
```

26 }；

## Input_Item 类

用于存储 input 参数的信息，每一个参数用 input_item 的一个对象存储，不同input_item 用 vector 打包。

1 class Input_Item   
2{   
3 public:   
4 Input_Item(const std::string& label_in)   
5 {   
6 label = label_in;   
7 }   
8   
9 std::string label; ///< label of the input item   
10 std::vector<std::string> str_values; ///< string values of the input item   
11 std: :stringstream final_value; ///< final value for writing to output INPUT file   
12 std::string annotation; ///< annotation of the input item   
13   
14 // ====== !!! These functions are complete. ======   
15 // ====== !! Do not add any more functions here. ======   
16 /// read value if INPUT file has this item   
17 std::function<void(const Input_Item&, Parameter&)> read_value = [](const Input_Item& item,   
18 Parameter& param) {};   
19 /// check value   
20 std::function<void(const Input_Item&， const Parameter&)> check_value = nullptr;   
21 /// reset some values   
22 std::function<void(const Input_Item&, Parameter&)> reset_value = nullptr;   
23 /// get final_value function for output INPUT file   
24 std::function<void(Input_Item&, const Parameter&)> get_final_value = nullptr;   
25 // ====== !!! Do not add any more functions here. ======   
26 }；

四个存储的变量：

label:参数的名字

annotation:参数的注释，用于生成INPUT的注释

str_values:读人的raw 数据，一个参数可以有多个数据，因此用vector存储，例如：kspacing 33 3

final_value:最终的取值，用于生成INPUT

四个 std::function 成员:

read_value（必填）：当从INPUT读入参数后，需要进行的赋值操作

reset_value（选填）：根据其读人的值，可能会修改其他参数的值时填入，例如calculation默认是 scf，当你读入calculation 是nscf后，则需要将init_chg改成file

check_value（选填)：根据其读人的值进行判断参数是否合适，例如：读入ecut 为-100，则需要 warning_quitO

get_final_value（大部分必填)：此函数一定会执行（无论是否从INPUT读入这个参数)，需要在此函数中给出final_value 的赋值函数，这样用于打印INPUT文件

## ReadInput 类

此类只负责读入INPUT文件，并赋值给Parameter类的对象，此类不负责存储参数

```cpp
1 class ReadInput
2{
3 public:
4 /**
5 * @brief read in parameters from input file
6 *
* @param param parameters of ABACUS
8 * @param filename_in read INPUT file name
9 */
10 void read_parameters(Parameter& param, const std::string& filename_in);
11
12 /**
13 * @brief write out parameters to output file
14 *
15 * @param param parameters of ABACUS
16 * @param filename_out write output file name
17 */
18 void write_parameters(const Parameter& param, const std::string& filename_out);
19 private:
20 std::vector<std::pair<std::string, Input_Item>> input_lists;
21 //----These functions are done only when INPUT file has them.----
22 // read value if INPUT file has this item
23 std::vector<Input_Item*> readvalue_items;
24
25 /// bcast all values function
26 /// if no MPI, this function will resize the vector
27 std::vector<std::function<void(Parameter&)>> bcastfuncs;
```

## 主要两个 public 函数:

write_parameters:打印 INPUT 文件

## 3个 vector:

bcastfuncs:用于所有参数的bcast

原则上 bcastfuncs 也可以用 vector 存储，一方面由于其不依赖 Input_Item，并且可能会bcast非 input 列表参数，另一方面但为了访存效率，用vector存储更好

## read_parameters 函数

对于ReadInput 的 read_parameter，其执行的顺序是:  
先确定 INPUT对应哪些 Input_Item，然后将 INPUT中读入的信息存人 item 的 str_values 中，之后依次执行  
1.readvalue_items  
2.inputlist->resetvalue  
3.inputlist->checkvalue  
4.bcastfuncs

write_parameters 函数  
如果要打印INPUT，顺序执行 input_lists 的 getfinalvalue  
然后

```cpp
1 for (auto& item: this->input_lists)
2{
3 Input_Item* p_item = &(item.second);
4 if (p_item->getfinalvalue == nullptr)
5 continue;
6 p_item->getfinalvalue(*p_item, param);
ModuleBase::GlobalFunc:0UTP(ofs, p_item->label, p_item->final_value.str()， p_item->annotation);
8}
```

## 7.8如何在ABACUS中进行异构计算

## 7.8.1 前言

异构编程（Heterogeneous Programming）是一种利用不同类型计算单元协同工作来完成计算任务的编程模式。传统的同构系统中，所有计算任务都由单一类型的处理器（如CPU）完成。但随着计算需求的不断增长，单一类型处理器在处理某些特定任务时效率有限。例如，CPU 虽然具有强大的通用性和控制能力，但在处理大规模并行计算任务时，性能表现不如专门的计算单元。因此，异构系统应运而生，它结合了不同类型的计算单元，如CPU、GPU（图形处理器）、FPGA（现场可编程门阵列)、ASIC（专用集成电路）等，以充分发挥各计算单元的优势，提高计算效率，这里我们主要讨论CPU 结合GPU（或者类似GPU）的异构编程。

在国产密度泛函理论软件ABACUS（原子算筹）里，异构的主要对象之一是数学库。近期我们对数学库blas的调用进行了封装，相关文件我们称为blas_connector。blas_connector得到重构之后（在这个PR 得到接收之后https://github.com/depmodeling/abacus-develop/pull/5799），可以通过直接调用 blas_connector.h 里面提供的算子，配合 device 和 memory 等基础模块即可直接编写异构代码。这套方案比起现在使用的方案来说，希望解决不同模块各自封装的问题，且对于新开发者来说可以避免使用模板和各自封装导致的理解困难。之后会逐渐使用这套方案替换原有的实现，从而实现整体异构方案的统一化与重构。用这种方法去除模板之后，能实现更灵活的计算，后续会用这个方式完成混合精度计算功能的实现。

下文将会介绍如何使用ABACUS 现有的工具简单的进行异构计算的实现。

## 7.8.2 管理设备类型

ABACUS 目前使用这几种方式进行计算设备的标识。主要记录在 source/module_base/module_device/types.h文件中。包括 struct 和enum 两种方式

```c
1 // !source/module_base/module_device/types.h
2 #ifndef MODULE_TYPES_H_
3 #define MODULE_TYPES_H_
4
5 namespace base_device
6 {
8 struct DEVICE_CPU;
9 struct DEVICE_GPU;
10 struct DEVICE_DSP;
```

```c
11
12 enum AbacusDevice_t
13{
14 UnKnown,
15 CpuDevice,
16 GpuDevice,
17 DspDevice
18 }；
19
20 } // namespace base_device
21
22 #endif // MODULE_TYPES_H_
```

其中，DEVICE_XXX是定义为结构体的标识符。这些结构体被作为模板参数使用，在之前是作为ctx 这种标识符使用，使用方式有些反直觉。

另一个 AbacusDevice_t就是一个枚举类型变量。是一个很早就出现在 ABACUS 的代码中但是几乎一直没有被使用的数据结构。在重构 blas_connector之后会逐渐启用，用于提供一个更加简单清晰的设备标识。

除此之外，在ABACUS里的有些其他模块会存在自己的设计。如source/mod-ule_basis/module_pw/module_fft/fft_base.h的 fft模块中，使用一个字符串来标记设备类型

1 std:string device = "cpu"; //或者是 gpu

这个类简单的使用一个字符串来标记device 类型。本质上和使用 AbacusDevice_t没有区别。这里建议新开发者如果希望加入新的异构功能，尽量都是用AbacusDevice_t来标记。

## 7.8.3 数据内存管理（旧)

由于这些Operator 的实现存在一些冗余的参数，因此会在近期的 PullRequest 中迎来对应的重构工作。因此暂时不建议使用这里的Operator进行内存管理，但是其仍然具有一定参考价值。

## 给指针分配内存

注：如果你需要进行大量数据的计算和转移（如需要储存一个矩阵)，这里建议使用ABACUS 提供的 Tensor数据结构进行计算，见Tensor文档。你也可以用这种方式进行异构计算，它会更加灵活，但是你需要自己进行内存管理，防止内存溢出等问题。

现在可用的异构内存工具位于 source/module_base/module_device/memory_op.h 文件中。只需要包含这个头文件即可使用。目前这些算子的使用方式还是以模板为主。我们主要会使用以下Operator。

```cpp
1 template <typename FPTYPE, typename Device>
2 struct resize_memory_op
3{
4 /// @brief Allocate memory for a given pointer. Note this op will free the pointer first.
5 //1
6 /// Input Parameters
/// \param size : array size
8 /// \param record_string : label for memory record
9 //
10 /// Output Parameters
```

```c
11 /// \param arr : allocated array
12 void operator()(const Device* dev， FPTYPE*& arr， const size_t size,
13 const char* record_in = nullptr);
14 }；
```

这个算子的主要作用是给一个指定指针分配内存。你也可以使用这个算子给以及包含了一段数据的指针分配内存，这样的话原来的内存区域会被释放，然后重新分配一段内存。你可以这样使用这个算子

```cpp
1 double* ptr = nullptr;
2 base_device::memory::resize_memory_op<double, base_device:DEVICE_CPu>()(ptr, sizeof(double));
```

可以注意到，首先我们有两个模板参数叫 base_device:DEVICE_GPU 和base_device:DEVICE_CPU，用于决定这个Operator会执行在哪个设备上。。

之后我们只要调用这个Operator来实现在不同设备分配内存。上文展示了在CPU的一个double指针上分配一定内存的方法。这个Operator的用法是这样的

1 base_device::memory::resize_memory_op<TYPE, DEVICE>()(pointer, size);

在尖括号模板参数中，TYPE 是需要分配的数据类型，如float，double，std:complex<double>等。DEVICE决定空间会被分配到哪个计算设备上，可以选填 base_device:DEVICE_CPU或 base_device::DEVICE_GPU。在后面的参数中，pointer 是需要被分配内存的指针，size 是希望分配的空间大小，一般可以用 sizeof(TYPE) \* count的方式指定分配多少个数据。

## 初始化一片内存的值

memory_op.h提供了以下算子来初始化一片内存区域的值

```c
template <typename FPTYPE, typename Device>
2 struct set_memory_op
3{
4 /// @brief memset for multi-device
5 ///
6 // Input Parameters
7 /// \param var : the specified constant value
8 /// \param size : array size
9 ///
10 /// Output Parameters
11 /// \param arr : output array initialized by the input value
12 void operator()(FPTYPE* arr， const int var，const size_t size);
13 }；
```

同样的，可以这样使用这个Operator

```rust
1 base_device::memory::set_memory_op<TYPE, DEVICE>()(pointer, var， size)
```

TYPE：被操作指针的数据类型

DEVICE：计算设备类型，可选base_device::DEVICE_CPU或 base_device::DEVICE_GPU

pointer：被操作的指针

var：设置内存区域的值，只能为0或者-1

size：设置内存区域的大小

注意！由于底层实现的原因，这个算子只能用来将一片内存区域全部设置为0或者全部设置为-1，不能用于给这篇内存填充别的特定数值。一般情况下我们也不会需要使用这个来填充别的数值的使用场景。

## 在不同设备间转移数据

memory_op.h提供了以下算子来在不同设备间转移数据

```c
1 template <typename FPTYPE, typename Device_out， typename Device_in>
2 struct synchronize_memory_op
3{
4 /// @brief memcpy for multi-device
5 ///
6 /// Input Parameters
/// \param arr_in : input array
8 /// \param size : array size
9 ///
10 /// Output Parameters
11 /// \param arr_out : output array initialized by the input array
12 void operator()(FPTYPE* arr_out,
13 const FPTYPE* arr_in,
14 const size_t size);
15 }；
16
17 template <typename FPTYPE_out， typename FPTYPE_in， typename Device_out， typename Device_in>
18 struct cast_memory_op
19{
20 /// @brief memcpy for multi-device
21 ///
22 /// Input Parameters
23 /// \param arr_in : input array
24 /// \param size : array size
25 //
26 /// Output Parameters
27 /// \param arr_out : output array initialized by the input array
28 void operator()(FPTYPE_out* arr_out,
29 const FPTYPE_in* arr_in,
30 const size_t size);
31}；
```

这个算子的作用是在不同计算设备间转移数据。其中 synchronize_memory_op 会转移同种数据类型的数据，cast_memory_op可以在设备间转移不同类型的数据，但是效率可能稍低。这个算子可以这么使用

1 base_device::memory::synchronize_memory_op<TYPE, DEVICE_OUT, DEVICE_IN>   
2 (pointer_out, pointer_in, size);

这个函数会将 pointer_in 长度为 size 的内容拷贝到 pointer_out。要求这两个指针的数据类型TYPE 要相同。如果选在相同的DEVICE_OUT和DEVICE_IN，会在同一个计算设备上的两个指针之间移动数据，也就是起到

一个拷贝的作用。 ^拷贝的作用。

TYPE：被操作指针的数据类型  
DEVICE_OUT：拷贝的目标计算设备类型，可选 base_device:DEVICE_CPU或 base_device:DEVICE_GPU  
DEVICE_IN：被拷贝的计算设备类型，可选 base_device:DEVICE_CPU或base_device:DEVICE_GPU  
pointer_in：被拷贝的指针  
pointer_out：拷贝的目标指针  
size：拷贝内存区域的大小  
另一个算子使用方法类似，但是会提供一个类型转换

1 base_device::memory::cast_memory_op<TYPE_OUT， TYPE_IN， DEVICE_OUT， DEVICE_IN>   
2 (pointer_out, pointer_in, size);

TYPE：被操作指针的数据类型

DEVICE_OUT：拷贝的目标计算设备类型，可选 base_device:DEVICE_CPU或 base_device::DEVICE_GPU

DEVICE_IN：被拷贝的计算设备类型，可选 base_device::DEVICE_CPU或 base_device::DEVICE_GPU

pointer_in：被拷贝的指针

pointer_out:拷贝的目标指针

size：拷贝内存区域的大小

使用这两个算子可以将CPU上的数据拷贝到GPU上，或者将GPU数据转移到CPU上。也可以在同一设备上使用实现内存区域的拷贝。

## 删除内存区域

为了防止内存溢出，一定要记得在开辟内存区域最后删除。在memory_op.h提供了以下算子来释放内存空间

```c
1 template <typename FPTYPE， typename Device>
2 struct delete_memory_op
3{
4 /// @brief free memory for multi-device
5 ///
6 /// Input Parameters
7 /// \param arr : the input array
8 void operator()(const Device* dev， FPTYPE* arr);
9}；
```

这个使用方式非常简单，只需要

```rust
1 base_device::memory::delete_memory_op<TYPE, DEVICE>()(pointer);
```

TYPE：被操作指针的数据类型

DEVICE：计算设备类型，可选 base_device::DEVICE_CPU或 base_device:DEVICE_GPU

pointer：需要删除内存区域的指针

## 使用别名

在一般使用的时候，为了避免算子名称过长导致的代码可读性下降，我们一般使用关键字using来缩短对应模板函数的名称，如

1 using resmem_sh_op = base_device::memory::resize_memory_op<float, base_device:DEVICE_CPU>;   
3 // Then you can directly using resmem_sh_op to resize memory

在 memory_op.h中，已经做好了大量这种定义

using resmem_sh_op = base_device:memory::resize_memory_op<float, base_device::DEVICE_CPU>;   
2 using resmem_dh_op = base_device::memory::resize_memory_op<double, base_device::DEvICE_CPU>;   
3 using resmem_ch_op = base_device::memory::resize_memory_op<std::complex<float>,   
4 base_device: :DEVICE_CPU>;   
5 using resmem_zh_op = base_device::memory::resize_memory_op<std::complex<double>,   
6 base_device: :DEVICE_CPU>;   
8 using resmem_sd_op = base_device::memory::resize_memory_op<float,   
9 base_device: :DEVICE_GPU>;   
10 using resmem_dd_op = base_device:memory::resize_memory_op<double, base_device::DEVICE_GPU>;   
11 using resmem_cd_op = base_device::memory::resize_memory_op<std: :complex<float>   
12 ， base_device::DEVICE_GPU>;   
13 using resmem_zd_op = base_device::memory::resize_memory_op<std::complex<double>,   
14 base_device: :DEVICE_GPU>;   
15   
16 using setmem_sh_op = base_device::memory::set_memory_op<float,base_device::DEVICE_CPU>;   
17 using setmem_dh_op = base_device::memory::set_memory_op<double,base_device:DEVICE_CPU>;   
18 using setmem_ch_op = base_device::memory::set_memory_op<std::complex<float>,   
19 base_device::DEVICE_CPU>;   
20 using setmem_zh_op = base_device::memory::set_memory_op<std::complex<double>,   
21 base_device: :DEVICE_CPU>;   
22   
23 using setmem_sd_op = base_device::memory::set_memory_op<float, base_device::DEVICE_GPU>;   
24 using setmem_dd_op = base_device:memory::set_memory_op<double， base_device::DEvICE_GPU>;   
25 using setmem_cd_op = base_device::memory::set_memory_op<std::complex<float>,   
26 base_device: :DEVICE_GPU>;   
27 using setmem_zd_op = base_device::memory::set_memory_op<std::complex<double>,   
28 base_device: :DEVICE_GPU>;   
29   
30 using delmem_sh_op = base_device::memory::delete_memory_op<float， base_device:DEVICE_CPU>;   
31 using delmem_dh_op = base_device:memory::delete_memory_op<double, base_device::DEVICE_CPU>;   
32 using delmem_ch_op = base_device::memory::delete_memory_op<std::complex<float>,   
33 base_device: :DEVICE_CPU>;   
34 using delmem_zh_op = base_device::memory::delete_memory_op<std: :complex<double>,   
35 base_device::DEVICE_CPU>;   
36   
37 using delmem_sd_op = base_device::memory::delete_memory_op<float, base_device::DEVICE_GPU>;   
38 using delmem_dd_op = base_device:memory::delete_memory_op<double, base_device:DEVICE_GPU>;   
39 using delmem_cd_op = base_device::memory::delete_memory_op<std::complex<float>,   
40 base_device: :DEVICE_GPU>;   
41 using delmem_zd_op = base_device:memory::delete_memory_op<std::complex<double>,   
42 base_device: :DEVICE_GPU>;

44 using syncmem_s2s_h2h_op 45 = base_device::memory::synchronize_memory_op<float, base_device:DEVICE_CPU, 46 base_device::DEVICE_CPU>;

47 using syncmem_s2s_h2d_op 48 = base_device::memory::synchronize_memory_op<float, base_device::DEVICE_GPU, 49 base_device: :DEVICE_CPU>;

50 using syncmem_s2s_d2h_op 51 = base_device::memory::synchronize_memory_op<float, base_device::DEVICE_CPU, 52 base_device: :DEVICE_GPU>;

53 using syncmem_d2d_h2h_op 54 = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, 55 base_device: : DEVICE_CPU>;

56 using syncmem_d2d_h2d_op 57 = base_device::memory: :synchronize_memory_op<double, base_device: :DEvICE_GPU, 58 base_device: :DEVICE_CPU>;

59 using syncmem_d2d_d2h_op 60 = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, 61 base_device: :DEVICE_GPU>;

63 using syncmem_c2c_h2h_op 64 = base_device::memory::synchronize_memory_op<std::complex<float>, base_device::DEVICE_CPU, 65 base_device:DEVICE_CPU>;

66 using syncmem_c2c_h2d_op 67 = base_device::memory::synchronize_memory_op<std:complex<float>, base_device:DEVICE_GPU, 68 base_device:DEVICE_CPU>;

69 using syncmem_c2c_d2h_op 70 = base_device::memory::synchronize_memory_op<std::complex<float>, base_device::DEVICE_CPU, 71 base_device: :DEVICE_GPU>;

72 using syncmem_z2z_h2h_op = base_device:memory::synchronize_memory_op<std:complex<double>, base_device:DEVICE_CPU, 74 base_device::DEVICE_CPU>;

75 using syncmem_z2z_h2d_op 76 = base_device:memory:synchronize_memory_op<std::complex<double>, base_device:DEVICE_GPU, 77 base_device: DEVICE_CPU>;

78 using syncmem_z2z_d2h_op 79 = base_device:memory:synchronize_memory_op<std::complex<double>, base_device:DEVICE_CPU, 80 base_device:DEVICE_GPU>;

82 using castmem_s2d_h2h_op 83 = base_device::memory::cast_memory_op<double, float, base_device::DEVICE_CPU, 84 base_device:DEVICE_CPU>;

85 using castmem_s2d_h2d_op 86 = base_device::memory::cast_memory_op<double， float， base_device::DEVICE_GPU, 87 base_device: :DEVICE_CPU>;

88 using castmem_s2d_d2h_op 89 = base_device:memory:cast_memory_op<double, float， base_device::DEvICE_CPU, 90 base_device: :DEVICE_GPU>;

```asm
91 using castmem_d2s_h2h_op
92 = base_device::memory::cast_memory_op<float， double, base_device::DEVICE_CPU,
93 base_device: :DEVICE_CPU>;
94 using castmem_d2s_h2d_op
95 = base_device::memory::cast_memory_op<float， double, base_device::DEVICE_GPU,
96 base_device: :DEVICE_CPU>;
97 using castmem_d2s_d2h_op
98 = base_device::memory::cast_memory_op<float， double, base_device::DEVICE_CPU,
99 base_device: :DEVICE_GPU>;
100
101 using castmem_c2z_h2h_op = base_device::memory::
102 cast_memory_op<std: :complex<double>， std: :complex<float>， base_device::DEVICE_CPU,
103 base_device: :DEVICE_CPU>;
104 using castmem_c2z_h2d_op = base_device::memory::
105 cast_memory_op<std::complex<double>， std::complex<float>， base_device::DEVICE_GPU,
106 base_device: :DEVICE_CPU>;
107 using castmem_c2z_d2h_op = base_device::memory::
108 cast_memory_op<std::complex<double>， std::complex<float>, base_device::DEVICE_CPU,
109 base_device: :DEVICE_GPU>;
110 using castmem_z2c_h2h_op = base_device::memory::
111 cast_memory_op<std::complex<float>， std::complex<double>， base_device::DEVICE_CPU,
112 base_device: :DEVICE_CPU>;
113 using castmem_z2c_h2d_op = base_device::memory::
114 cast_memory_op<std: :complex<float>， std: :complex<double>， base_device::DEVICE_GPU,
115 base_device: :DEVICE_CPU>;
116 using castmem_z2c_d2h_op = base_device::memory::
117 cast_memory_op<std::complex<float>， std::complex<double>， base_device::DEVICE_CPU,
118 base_device: : DEVICE_GPU>;
```

在这个命名中，s,d,c,z 分别代表 float,double,std:complex<float>,std::complex<double>这四种数据类型。而h代表host，一般都指的是CPU，d代表device，一般都指GPU。

所以一个名为 syncmem_s2s_h2d_op的别名指的是一个将float转换为float（s2s),将数据从CPU拷贝到GPU(h2d）的 synchronize_memory_op。

## 7.8.4数据内存管理(新)

在Pull Request htps:/github.com/deepmodeling/abacus-develop/pul/5861 合并之后，添加了直接进行内存管理的函数。这些函数本质上是对上述算子的封装，但是使用它可以使代码更加清晰，且避免了增加重构工作的工作量。因此建议使用这些函数进行内存管理。

## 给指针分配内存

```cpp
1 template <typename FPTYPE>
2 void resize_memory(FPTYPE* arr, const size_t size, base_device::AbacusDevice_t
3 device_type = base_device::AbacusDevice_t::CpuDevice);
```

直接调用这个函数即可给指针arr分配内存。FPTYPE可为包括float，double，std:complex和std:complex的任意数据类型。device_type可选base_device::AbacusDevice_t::CpuDevice或者

base_device:AbacusDevice_t:GpuDevice。

## 初始化一片内存的值

```c
template <typename FPTYPE>
2 void set_memory(FPTYPE* arr, const int var, const size_t size, base_device::AbacusDevice_t
3 device_type = base_device::AbacusDevice_t: :CpuDevice);
```

直接调用这个函数即可给指针arr管理的一片内存区域初始化值。

其中，var 是需要赋的数值，只能为0 或者-1！ size 是这片内存区域的长度。device_type 可选base_device:AbacusDevice_t:CpuDevice 或者 base_device:AbacusDevice_t:GpuDevice。

注意！由于底层实现的原因，这个算子只能用来将一片内存区域全部设置为0或者全部设置为-1，不能用于给这篇内存填充别的特定数值。一般情况下我们也不会需要使用这个来填充别的数值的使用场景。

## 在不同设备间转移数据

有如下两个函数可以用于在不同设备间转移数据。

```c
1 template <typename FPTYPE>
2 void synchronize_memory(FPTYPE* arr_out， const FPTYPE* arr_in, const size_t size, base_device::
3 AbacusDevice_t device_type_out， base_device::AbacusDevice_t device_type_in);
4
5 template <typename FPTYPE_out， typename FPTYPE_in>
6 void cast_memory(FPTYPE_out* arr_out， const FPTYPE_in* arr_in, const size_t size, base_device:
7 AbacusDevice_t device_type_out， base_device::AbacusDevice_t device_type_in);
```

其中，synchronize_memory 负责转移同类型数据，cast_memory 可以转移不同类型的数据。

arr_out 是转移数据的目标指针，arr_in 是转移数据的原始指针。size 是转移数据的长度。

device_type_out是目标设备类型，可选base_device:AbacusDevice_t::CpuDevice或者base_device:AbacusDevice_t:GpuDevice。

device_type_in 是数据原本储存区域的设备类型，可选 base_device::AbacusDevice_t:CpuDevice 或者base_device:AbacusDevice_t:GpuDevice 。

如果device_type_out 和device_type_in 相同的话，则是进行一次简单的数据拷贝工作。

## 删除内存区域

为了防止内存溢出，一定要记得在开辟内存区域最后删除。只需要简单的执行这个函数即可

```cpp
1 emplate <typename FPTYPE>
2 void delete_memory(FPTYPE* arr, base_device::AbacusDevice_t device_type =
3 base_device:: AbacusDevice_t: :CpuDevice);
```

这个函数会删除指定设备上的arr指针的内存区域。

## 7.8.5执行数学计算

现在可以直接调用 Blas_connector进行异构计算。只需要包含 source/module_base/blas_connector.h 这个头文件即可。

在blas_connector.h中，链接了blas 的诸多函数（不建议直接使用)，且都实现了对应的异构算子，只需要调整一个参数即可选择对应的异构设备进行计算。

要使用实现好的算子，只需要调用BlasConnector命名空间下的算子即可。具体有哪些算子直接查看 source/-module_base/blas_connector.h。以最常用的 gemm 为例，这些算子一般是这么构成的

```c
1 static
2 void gemm(const char transa, const char transb, const int m, const int n, const int k,
3 const float alpha，const float *a, const int lda，const float *b, const int ldb,
4 Const float beta， float *c， const int ldc， base_device::AbacusDevice_t device_type =
5 base_device: : AbacusDevice_t: :CpuDevice);
6
7 static
8 void gemm_cm(const char transa, const char transb, const int m, const int n, const int k,
9 const float alpha， const float *a, const int lda, const float *b,const int ldb,
10 const float beta, float *c， const int ldc， base_device::AbacusDevice_t device_type =
11 base_device: : AbacusDevice_t: :CpuDevice);
```

这个算子前面的所有参数都和标准 blas 算子的参数排列方式相同，只要以相同的方式传人参数即可。注意到最后一个参数是 base_device:AbacusDevice_t类型的参数，这是一个来自上文所说的 source/mod-ule_base/module_device/types.h 的枚举类型，代表使用的计算设备。如果我们需要在GPU上调用对应的算子进行计算的话，就这么使用

```rust
1 BlasConnector::gemm(...parameters...， base_device::AbacusDevice_t::GpuDevice);
```

如果需要在 CPU上进行计算，只需要把这个参数改成 base_device:AbacusDevice_t::GpuDevice 或者直接空着不写（不建议)，因为默认值就是CPU参数。

可以注意到，对于相同的算子有时候会有两种类型的封装，其中一种有后缀_cm，这代表这是一个列优先的封装，如果没有后缀则为一个行优先的封装。由于blas 本身都是基于列优先实现的，所以部分算子可能没有行优先的封装。

这样你就可以使用上文所述的内存管理工具和 Blas 计算库，在完全不使用专用的异构API，如CUDA，来进行异构代码的编写，实现功能在不同的设备上都能够运行。

## 7.9ABACUS formatter-2.0 版本使用说明书

## 7.9.1 前言

为系统解决 ABACUS 对输出可读内容到屏幕和文件中的需求，2023第三季度 ABACUS开发团队进行了ABACUS formatterlibrary1.0版本的开发。随着后期 ABACUS 各功能的输出需求对 formatter library 的功能要求越来越多样化,我们对 formatterlibrary代码进行了全面重构,推出了formater-2.0版本。相较于1.0版本（ABACUSformatter 库使用说明书)，重构进行了大量冗余代码删减，如今已经轻量化成为只具有头文件、有明显“即插即用”性质的工具库。该工具库功能包括简单的字符串格式化、制表和部分Python字符串处理函数实现三部分。

## 7.9.2 字符串格式化

相较于1.0版本中只能对字符串加以1)宽度、2)小数保留位数、3)正负号、4)左右对齐和 5)科学计数法的设定，2.0版本基于 std:snprintf 函数（见cppreference：https://en.cppreference.com/w/cpp/io/c/fprintf）实现了对format string（例如 C++20 支持的 formatlibrary:htps:/en.cppreference.com/w/cpp/utility/format/format）的全面支

持（如%d,%o,%x,%u,%hd,%ld,%lu,%c,%s,%f,%e 等），弃置了基于 stream (iostream,）的技术路线（见https://stackoverflow.com/questions/15106102/how-to-use-c-stdostream-with-printf-like-formating）。

另一方面，std:cout 是在 ABACUS 中最普遍出现，且经常被修改的全局变量。任何不刻意设置便输出的内容都将受到“上一次”对 std:cout 的 IOManipulator 的影响。

然而，使用新版的 formatter 需要注意数据类型，若数据类型出现不匹配，则很可能出现undefined behavior（例如issue #4540 中所报道），这一类型要求和Python 自 3.12 版本开始功能健全且大范围流行和提倡的f-string一致。

## static format函数的使用

实际上，由于 formatter library 的轻量化重构，从单元测试中就足以可以明白 formater library这一基础函数的使用方法。又因为该函数是类中 static函数，因此可以随时避免创建类对象而直接使用该函数。基本的使用方法展示在下面：

```cpp
TEST(FormatterTest， FmtCoreStaticFormat) {
2 // const char*
3 std::string result = FmtCore::format("Hello, \%s!", "world");
4 // remove the last '\o' character
5 EXPECT_EQ(result， "Hello， world!");
6 // std::string
7 result = FmtCore::format("Hello, \%s!", std::string("world"));
8 EXPECT_EQ(result， "Hello，world!");
9 // int
10 result = FmtCore::format("Hello， \%d!"，123);
11 EXPECT_EQ(result， "Hello，123!");
12 // float
13 result = FmtCore::format("Hello, \%f!", 123.456);
14 EXPECT_EQ(result， "Hello， 123.456000!");
15 // char
16 result = FmtCore::format("Hello，\%c!"，'a');
17 EXPECT_EQ(result， "Hello， a!");
18 // invalid format
19 result = FmtCore::format("Hello, \%z!", "world");
20 EXPECT_EQ(result， "Hello，\%!");
21 // varadic template case
22 result = FmtCore::format("Hello， \%s， \%d, \%f， \%c!"， "world"， 123，123.456，'a');
23 EXPECT_EQ(result， "Hello，world，123，123.456000，a!");
24 }
```

下面我们对 ABACUS 中常用于输出的数据类型进行详细举例。int类型输出：%d

```cpp
1 #include "module_base/formatter.h"
2 #include <iostream>
3
4 std::cout << FmtCore::format("\%d"， 1);
5
6 //output: "1"
```

如果想要修改其宽度，则需要在d之前设定数字，如设置宽度为4:

```rust
1 std::cout << FmtCore::format("\%4d", 1);
2 //output: "1"
```

如果需要输出左对齐而非右对齐的字符串，则在百分号%之后添加负号：

```cpp
1 std::cout << FmtCore::format("\%-5d", 10);
2 // output: "10 =
```

float/double类型输出：%f和%e

对于浮点型数据，最重要的是其是否使用科学计数法、保留位数和宽度各如何。请认真观察如下示例：

```cpp
1 const double rough_pi = 3.1415926535897932384;
2 std::cout << FmtCore::format("\%8f"，rough_pi);
3 // output: "3.141592"
std::cout << FmtCore::format("\%8.4f", rough_pi);
5 // output: " 3.1415"
6 std::cout << FmtCore::format("\%.4f", rough_pi);
// output: "3.1415"
8 std::cout << FmtCore::format("\%.Of", rough_pi);
9 // output: "3"
```

不难发现如果f（或者e）前面只有一个数字，该数字默认为最小宽度。如果在数字前有小数点，则该数字意为保留小数位数，因此%.0f将直接对数字取整。

对于需要使用科学计数法的场景，只需要将f替换成e即可。需要注意的是科学计数法中诸如e+00也需要占位4个长度，需要在规定字符串输出长度时加以考虑。

## std::string类型输出：%s

对于字符串的输出最简单，另外一个具有相似功能的占位符为%c，意为为char类型在格式化字符串（formatstring）中占位。

```cpp
1 std::cout << FmtCore::format("Hello, \%s!\n", "world");
2 // output: "Hello， world!"
```

## dynamic format函数的使用

为应对大批量、重复使用同一format string的需求，可以首先建立一个FmtCore 对象，之后调用对象中的成员函数 format时，就不需要再每次输入 format string:

```cpp
1 FmtCore fmt("Hello, \%s!\n");
2 std::cout << fmt.format("world") << std::flush;
3 std::cout << fmt.format("again") << std::flush;
4 // output:
5 // "Hello， world!
6 // Hello,again!"
```

因此在功能角度和 static 函数中的 format基本无异。单元测试可以辅助理解：

```cpp
TEST(FormatterTest， FmtCoreDynamic)
2{
3 FmtCore fmt("Hello, \%s!");
4 EXPECT_EQ(fmt.fmt()，"Hello，\%s!");
5 std::string result = fmt.format(std::string("world"));
6 EXPECT_EQ(result， "Hello，world!");
8 fmt.reset("Hello， \%d!");
9 EXPECT_EQ(fmt.fmt()， "Hello，\%d!");
10 result = fmt.format(123);
11 EXPECT_EQ(result， "Hello，123!");
12
13 fmt.reset("Hello， \%f!");
14 EXPECT_EQ(fmt.fmt()，"Hello，\%f!");
15 result = fmt.format(123.456);
16 EXPECT_EQ(result， "Hello，123.456000!");
17
18 fmt.reset("Hello， \%c!");
19 EXPECT_EQ(fmt.fmt()，"Hello，\%c!");
20 result = fmt.format('a');
21 EXPECT_EQ(result， "Hello，a!");
22
23 // varadic template case
24 fmt.reset("Hello，\%s，\%d， \%f，\%c!");
25 EXPECT_EQ(fmt.fmt()，"Hello，\%s，\%d，\%f，\%c!");
26 result = fmt.format(std::string("world")，123，123.456，'a');
27 EXPECT_EQ(result， "Hello， world，123，123.456000，a!");
28}
```

## 7.9.3制表功能

formatter-1.0的另一项亮眼功能为自动制表，在 formatter-2.0版本中这一feature 得以保留，并由1.0版本的Table类重构成为更加轻量级的FmtTable类，用于提供想要制作排列整齐的数据表，又不太清楚表格的列宽的场景。

由于FmtTable 类的设计初衷为”每一个FmtTable instance“代表了一个Table，因此在 Table 中设计需要用户提供如下信息：

每列标题，组织成 std:vectorstd:string

每列 format string，组织成 std::vectorstd:string

每列数据

（可选）表中数据与表标题的左右对称

（可选）表格的各个边框

（可选）表格中每列的分隔符

## 形参表一览

```diff
1 /**
* @brief Construct a new Fmt Table object
3 *
4 * @param titles titles， its size should be the same as the number of columns
5 * @param nrows number of rows
6 * @param aligns Alignments instance, can be constructed with initializer_list<char> like
{'r'，'c'}, for right and center alignment for values and titles
8 * @param frames Frames instance， can be constructed with initializer_list<char> like
9 {'-'，'-'，'-'，''，' '}, for up,middle,down， left and right frames
10 * @param delimiters Delimiters instance, can be constructed with initializer_list<char> like
11 {'-'，' '}，for horizontal and vertical delimiters
12 */
13 FmtTable(const std:vector<std::string>& titles,
14 const size_t& nrows,
15 const std::vector<std::string>& fmts,
16 const Alignments& aligns = {,
17 const Frames& frames = {,
18 const Delimiters& delimiters = {}): titles_(titles), fmts_(fmts)，data_(nrows,
19 titles.size())，aligns_(aligns)， frames_(frames)，delimiters_(delimiters)
20 { assert(titles.size() == fmts.size()); };
```

基于RAII原则，我们假设需要制表并进行输出的数据经常是已经全部准备好的状态，而非如同 SCF的迭代信息一样，需要特别估算每列的大致宽度信息。

因此在FmtTable 构造函数中，我们一定需要titles,n_rows,fmts，并对 align 和 Delimiter 参数设置了默认值因此并非总是需要。如果想要特殊配置，则可以选择单元测试中例子 htps:/github.com/deepmodeling/abacus-develop/blob/develop/source/module_base/test/formatter_test.cpp#L323，仿照其进行制表调整。

## 使用代码示例

除了单元测试(htps:/github.com/deepmodeling/abacus-develop/blob/develop/source/module_base/test/formatter_test.cpp）外，目前FmtTable 也已经用于 ABACUS运行时间统计的输出（htps:/github.com/deepmodeling/abacus-develop/blob/develop/source/module_base/timer.cpp#L280），但可进一步实现对表中数据区域的每列左右对齐控制。

## 7.9.4 static Python-style 字符串函数

## 函数功能简介

C/C++ 的文件读入功能也较为繁琐，尤其涉及需要进行字符串操作时，Python中 str的函数则更胜一筹。为了避免每次手动 parse 字符串，又注意到 std:strng同样是 STL的标准容器，因此对标准容器支持的算法，也都能支持std:string。目前出于个人使用习惯，实现了下列Python-style static 函数：

●split:用于以固定分隔符切割字符串，返回 std::vectorstd::string，但也是因为该返回类型，注定 split函数不会收录在 STL 中

· startswith/endswith:返回boolean，用于判断字符串是否以给定字符串开始，或者以给定字符串结束。

·strip:用于消除出现在字符串行头行尾的空格、回车和\o等字符，返回 std::string。

·center:为了支持下一步的 ABACUS 输出重构，首先实现center函数用于将字符串以某一个宽度居中，两段则填充以center 函数第一个参数，使得 ABACUS 中所有内容都可以具有给定宽度的输出（即两端对齐)

replace:用于消除字符串中存在的所有某个字符

join:split的反函数

## 代码示例

以上函数的例子可见单元测试:htps:/github.com/depmodeling/abacus-develop/blob/develop/source/module_base/test/format

## 7.9.5 Find a bug? Submit issue!

如果在使用 formatter 过程中发现了bug 或者运行结果不达预期，可以在deepmodeling/abacus-develop 仓库下提交 issue。

## 7.10ABACUS中使用格式化工具clang-format

## 7.10.1 引言

良好统一的代码风格在多人协同开发中至关重要，这一点在ABACUS 团队里也尤其重要，特别是考虑到大部分的开发者都是研究生，没有经过特别系统的编程训练，那么好的代码风格能够降低阅读代码的障碍，加快开发效率，甚至可以防止一些不必要的bug 的产生。

clang-format是一个开源的代码格式化工具，它可以帮助程序员自动调整源代码的格式，以符合指定的编码风格规范。clang-format可以在保持代码功能不变的情况下，自动处理缩进、空格、括号、逗号等细节，提高代码的可读性和一致性。

## 7.10.2 在ABACUS中使用clang-format

clang-format 支持包括LLVM、Google、Chromium、Mozilla、WebKit等在内的多种代码风格，如果不指定自定义的代码风格，每次对代码进行格式化时使用的将是clang-format 默认的风格。

clang-format还支持在已有的代码风格基础之上自定义一个名为.clang-format 的代码风格文件，clang-format就可以根据文件中定义的规则对代码进行格式化。

在 ABACUS 开源项目中，开发者已经制定了针对 ABACUS项目的代码风格文件.clang-format，其位于项目的根目录下。（https://github.com/deepmodeling/abacus-develop/blob/develop/.clang-format)

注意事项：ABACUS 开源项目中的.clang-format 是依赖于clang-format-14 标准的，因此请确保你的 clang-format不低于14版本。

## 在系统上安装clang-format

clang-format 有诸多版本可供使用，目前最新版本为clang-format 18。不同版本所支持的格式化选项不尽相同，但向后兼容。

安装统一版本的 clang-format（ABACUS项目中统一选择clang-format-14)

1 # Linux 上，以 Ubuntu 为例:  
2 sudo apt install clang-format-14  
3  
4#验证是否安装成功，并查看安装的版本号：  
5 clang-format-14 --version

这里应注意，不能通过 sudo apt installclang-format命令进行安装，因为对于不同的Ubuntu发行版而言，通过该命令安装得到的clang-format版本是不同的。

---

## 第 361–362 页

安装完成后，我们能够使用的命令名是clang-format-14，我们将其提级为clang-format:

1 sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 100

后面当我们使用clang-format命令时，指代的即为clang-format-14。

1# 此时再次验证安装的版本号：(应为 14 版本)  
2 clang-format --version

## 基于 vscode 使用clang-format

基于 vscode 使用clang-format，我们可以选择安装支持clang-format 的插件，即可更方便的使用clang-format。

但由于插件商场有很多插件都支持clang-format功能，且其支持的clang-format 版本经常变化。所以我们应尽量选择支持更为稳定的插件进行使用。

通过 vscode 进行C++开发时，我们一般会安装微软官方的C/C++插件，打开该插件的安装目录我们会发现，该插件已经封装了一个内置的clang-format。那为什么不直接使用这个C/C++插件内置的clang-format呢?不同版本的C/C++ 插件内置的clang-format 版本往往不同，就带来不必要的麻烦。为了使用统一的 clang-format-14，我们建议在 vscode 中安装 Clang-Format（Xaver Hellauer）插件。

安装完成后，打开“File -> Preferences -> Settings”，搜索“format”，如下图:

<!-- image-->

将Default Formatter 刚换为新下载的 Clang-Format 插件。

之后在一个文件内，通过"option”+"shift”+"F”快捷键，即可实现对当前代码文件的格式化。

## 基于命令行使用clang-format

除了使用vscode，可能还会有使用命令行进行开发的开发者。

使用方式：

```batch
1 clang-format -style=file -i demo.cpp
```

## 7.10.3格式化时的注意事项：

由于直接对整个文件格式化会导致每一行的历史修改信息被覆盖，因此我们要求在您修改某文件相应的代码内容之后，先commit一次，然后再对该文件进行格式化，再提交一次专门对文件格式化的commit。这样细化每一个commit具体的信息，更有利于PR 的审核和项目的管理。

## 7.10.4 参考文献:

使用clang-format 进行C++ 代码风格管理

Clang-Format用法详解

Clang-Format Style Options Clang 14.0.0 documentation

C++ programming with Visual Studio Code

基于vscode 使用clang-format （I旧版）

---
