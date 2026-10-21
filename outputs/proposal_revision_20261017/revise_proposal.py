from pathlib import Path
import re, math, json
import fitz
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

ROOT=Path('/root/DroneVisCrt')
OUT=ROOT/'outputs/proposal_revision_20261017'
SRC=ROOT/'无人机巡检多模态智能体开题报告_修订版.pdf'
doc=fitz.open(SRC)
original_blocks={i+1:doc[i].get_text('blocks') for i in range(43)}
# Recover Unicode cmaps from the source PDF's embedded fonts, preserving its typography.
for xref,label in [(7,'body'),(5,'heading')]:
 name,ext,typ,data=doc.extract_font(xref)
 raw=OUT/(label+'.ttf');raw.write_bytes(data)
 font=TTFont(raw)
 unicode_ref=int(re.search(r'/ToUnicode (\d+)',doc.xref_object(xref)).group(1))
 cmap_text=doc.xref_stream(unicode_ref).decode()
 pairs={}
 for section in re.findall(r'beginbfchar(.*?)endbfchar',cmap_text,re.S):
  for a,b in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>',section):pairs[int(b,16)]=int(a,16)
 for section in re.findall(r'beginbfrange(.*?)endbfrange',cmap_text,re.S):
  for a,b,c in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>',section):
   for gid in range(int(a,16),int(b,16)+1):pairs[int(c,16)+gid-int(a,16)]=gid
 cm=CmapSubtable.newSubtable(4);cm.platformID=3;cm.platEncID=1;cm.language=0
 cm.cmap={code:font.getGlyphName(gid) for code,gid in pairs.items() if code<=65535}
 font['cmap']=newTable('cmap');font['cmap'].tableVersion=0;font['cmap'].tables=[cm];font.save(OUT/(label+'_unicode.ttf'))
FONTS={'body':fitz.Font(fontfile=str(OUT/'body_unicode.ttf')),'heading':fitz.Font(fontfile=str(OUT/'heading_unicode.ttf')),'latin':fitz.Font('tiro'),'sans':fitz.Font('cjk')}
REGISTER={'body':'RBody','heading':'RHead','latin':'tiro','sans':'RSans'}
missing=set();log=[];queue={}; jobs=[]

def font_for(c,style):
 f=FONTS[style]
 if f.has_glyph(ord(c)):return style
 if ord(c)<256:return 'latin'
 missing.add(c);return 'sans'
def width(s,fs,style='body'):
 return sum(FONTS[font_for(c,style)].text_length(c,fontsize=fs) for c in s)
def put(page,x,y,s,fs=12,style='body',color=(0,0,0)):
 groups=[]
 for c in s:
  f=font_for(c,style)
  if groups and groups[-1][0]==f:groups[-1][1]+=c
  else:groups.append([f,c])
 for f,t in groups:
  page.insert_font(fontname=REGISTER[f],fontbuffer=FONTS[f].buffer)
  page.insert_text((x,y),t,fontname=REGISTER[f],fontsize=fs,color=color)
  x+=width(t,fs,f)

def wrap(text,w,fs=12,style='body',indent=0):
 lines=[];line='';available=w-indent
 for c in text:
  if c=='\n':lines.append(line);line='';available=w;continue
  if width(line+c,fs,style)>available and line:
   if c in '，。；：、！？）】》”％':line+=c;lines.append(line);line=''
   else:lines.append(line);line=c
   available=w
  else:line+=c
 if line:lines.append(line)
 return lines

def paragraph(page,rect,text,fs=12,indent=24,leading=20.5,style='body',justify=True):
 r=fitz.Rect(rect)
 while True:
  lines=wrap(text,r.width,fs,style,indent)
  if (len(lines)-1)*leading+fs*1.12<=r.height+0.6:break
  fs-=0.15
  if fs<10:raise ValueError(('Paragraph does not fit',page.number+1,rect,text))
 # SimSun ascender .86; match source baseline.
 baseline=r.y0+fs*0.90
 for i,line in enumerate(lines):
  x=r.x0+(indent if i==0 else 0)
  available=r.x1-x
  if justify and i<len(lines)-1 and len(line)>1:
   extra=(available-width(line,fs,style))/(len(line)-1)
   for c in line:
    put(page,x,baseline+i*leading,c,fs,style);x+=width(c,fs,style)+extra
  else:put(page,x,baseline+i*leading,line,fs,style)
 log.append({'page':page.number+1,'font_size':round(fs,2),'lines':len(lines),'text':text})

def erase(n,rect,graphics=False):
 queue.setdefault(n,[]).append((fitz.Rect(rect),graphics))

def replace(n,start,end,text,indent=24,fs=12):
 bs=original_blocks[n][start:end+1]
 r=fitz.Rect(73.6,min(b[1] for b in bs)-0.4,531,max(b[3] for b in bs)+0.6)
 erase(n,r);jobs.append(lambda n=n,r=r,text=text,indent=indent,fs=fs:paragraph(doc[n-1],r,text,fs,indent))

def line_replace(n,k,text,fs=11.95,style='body'):
 b=original_blocks[n][k];r=fitz.Rect(b[:4]);erase(n,r+(-.3,-.3,.3,.3))
 def run():
  size=min(fs,(r.width+2)/max(width(text,fs,style),1)*fs)
  if size<9.4:raise ValueError(('line too small',n,k,text,size))
  put(doc[n-1],r.x0,r.y1-2.3,text,size,style)
 jobs.append(run)

# Figure captions: remove parenthetical notes; retain source citations outside parentheses.
captions={1:'研究总体框架图',2:'场景级新视角生成框架[33]',3:'持续测试时适应框架[8]',4:'反思与经验记忆结构[64]',5:'可配置执行单元与会话编排[69]',6:'研究方案总体实施流程',7:'研究点一技术路线：目标相关视角生成与样本扩增',8:'视觉证据、区域关系与规则风险评估流程',9:'研究点二技术路线：先预测、后反馈、再更新的受约束自进化',10:'管理平台架构：界面、控制服务、独立工作进程和版本存储',11:'研究点三技术路线：版本化任务编排、证据归档与恢复回放',12:'统一实验闭环与标签隔离：开发反馈和最终测试分离'}
for n,blocks in original_blocks.items():
 for b in blocks:
  t=b[4].strip();m=re.match(r'图(\d+)\s*[:：]',t)
  if not m:continue
  num=int(m.group(1));erase(n,fitz.Rect(b[:4])+(-.3,-.3,.3,.3))
  text=f'图{num}: '+captions[num]
  def caption(n=n,b=b,text=text):
   fs=10.5;put(doc[n-1],(595.28-width(text,fs))/2,b[3]-3,text,fs)
  jobs.append(caption)

# Revise prescribed software/model/data choices into implementation-neutral technical plans.
replace(8,19,25,'检测方法从候选区域到端到端建模形成了多种路线。区域建议网络将候选框生成与检测关联起来[15]，适合分析候选区域与小目标召回之间的关系；统一预测框架体现了单阶段方法的执行效率优势[16]；集合预测与匹配损失则提供了减少人工后处理的另一条路线[17]。本课题拟选用具有公开权重、稳定实现和明确许可的目标检测模型建立基线，根据小目标表现及资源开销在开发集确定配置。实验前冻结实现版本、权重、训练参数和运行预算，保证各扩增方法在相同检测条件下比较[18]。')
replace(11,13,18,'国内团队发展的视觉语言模型进一步探索高分辨率视觉、定位、视频及多任务能力[46][47]。本课题拟选用具备公开权重和明确输入接口的视觉语言模型建立基线，并记录检查点与预处理配置[48]。运行时显式提供图像、任务和检测证据，模型负责受约束的工具策略与说明，最终风险等级由规则引擎计算。由于通用基准不能覆盖特定场景规则，需要建立专门的巡检任务集合，检验模型是否引用真实存在的目标与区域。')
replace(11,19,25,'参数高效适配是有限资源下的重要路径。可学习提示及输入相关的条件上下文为视觉语言任务适配提供了依据[49][50]，但不能直接当作生成式智能体的完整更新方案。低秩参数增量可替代全量权重更新[51]，结合低比特基座可进一步降低微调内存要求[52]。本课题拟采用轻量参数适配，在显存预实验后确定数值精度及更新配置，分别报告量化与非量化条件的开销。权重融合研究表明新任务收益与分布外泛化能力需要共同度量[53]。')
replace(15,11,16,'底层运行框架可以提高实验组织效率，但应与研究结论分开。任务与参与者抽象为分布式执行提供了组织方式[78]，分页式注意力缓存为语言模型服务的内存管理与吞吐优化提供了思路[79]。本课题以单卡顺序实验为基础，采用轻量工作进程与任务队列组织原型。模型生命周期管理研究强调训练、部署和监控之间的联系[80]，说明数据、代码、模型和运行版本的关联是持续迭代的必要条件。后续扩展吞吐时，再评价额外运行框架的收益与开销。')
replace(22,21,25,'研究对象为固定场景在不同相机条件下的视觉观测。拟以场景级新视角生成为基础，利用单张或稀疏参考图像构建粗几何条件，在受限相机轨迹上生成新的可见视图；重点研究巡检目标的投影尺度、可见性和视角覆盖如何影响采样选择。针对已有图像中检测困难的人员、车辆以及补充标注的障碍物，构造任务驱动的视角排序与固定预算样本选择策略，比较其与随机、均匀采样的差异。')
replace(23,1,5,'研究对象为由目标检测模块、视觉语言模型、确定性规则引擎与经验库组成的单智能体。拟首先建立可固定回放的任务策略：接收图像与区域配置，调用检测模块，关联目标与区域，执行风险规则，并输出含对象编号、区域编号、规则编号和证据位置的结构化报告。在此基础上，将失败分解为调用错误、证据缺失、输出格式错误、区域关系不确定及检测不足等类型，使可更新对象与失败来源相匹配。')
replace(23,12,16,'研究对象为实验任务从输入到更新再到回放的完整生命周期。拟建立任务、样本、标注、模型、适配器、规则和经验版本之间的显式关联，通过前端交互、任务控制服务、结构化存储和对象存储搭建原型，将生成、检测推理和适应训练放入独立工作进程。平台支持公开数据回放、任务配置、检测与风险证据查看、候选版本比较、运行轨迹存档及恢复。')
replace(24,15,20,'本课题拟采用“公开基线复现—受控方法改进—真实数据评价—系统闭环验证”的研究路线。首先建立数据及标注协议，复现场景级生成、目标检测和视觉语言理解基线，冻结模型及配置；随后研究目标相关视角选择，并将生成图仅加入训练域；在检测模块与规则固定的条件下比较智能体参数和记忆更新；最后将各方法接入同一平台，检验结果、资源与回放一致性。全部方法在开题阶段均为拟开展工作，所列超参数是实验起点，最终配置经开发集与资源预实验确定后冻结。')
replace(25,14,19,'（1）数据来源、划分与增量标注。生成验证选用具有相机参数与几何信息的公开合成城市场景数据，读取彩色图像、深度、内参及相机变换，并明确标注合成来源。真实检测使用公开航拍图像，风险与连续适应使用含道路、停车区域及园区形态的公开视频片段构建补充子集。场景筛选依据可见结构与数据说明，不推断场所身份。数据来源按许可、标注完整性和任务覆盖确定；对覆盖不足的场景保留独立验证需求。各类数据的职责见表4。',indent=17)
# Dataset names in table 4 are replaced as complete cells below.
replace(26,5,11,'真实检测保留所选公开数据的原始划分信息，按场景组建立训练集、开发集和冻结测试集；具有官方独立测试入口时，另作一次性检查。开发集按可确认的序列、采集位置及视觉近重复连通组留出20%，其余80%用于训练。对跨划分疑似同场景图像，结合文件来源、感知哈希候选和人工核对形成冲突清单，移除训练侧冲突组；无法确认时保守合并或排除，并报告数量。官方全量基准可作为单独对照，但其分数与严格隔离子集分别报告。')
replace(26,19,24,'人员、车辆及障碍物采用统一语义规范，并保留原始类别到研究类别的映射。将数据中语义一致的人员类别合并；车辆按任务需要区分主要机动车与其他车辆，分别报告，不随意删除已有标注形成负样本。障碍物定义为可见的、非通行参与者的实体阻挡物，包括临时路障、围挡、散落物与堆置物；常驻建筑和道路边缘不作为障碍物。对原数据未覆盖的障碍物实施完整增量标注，未完成补标的图像不进入该类训练负例。忽略区域及类别不确定区域使用掩码处理。')
replace(27,2,4,'（2）相机条件及未知位姿处理。内部统一采用右手相机坐标：x向右、y向下、z向前，以相机到世界的变换记录位姿。对于源像素齐次坐标及其沿相机光轴的深度，新视角投影为',indent=17)
replace(27,8,13,'其中，K、K′为两相机内参，R、t为相对旋转和平移，D为沿光轴的深度。像素坐标经齐次归一化获得投影位置，遮挡采用深度缓冲处理。不同来源的相机坐标约定、深度定义和长度单位在导入时统一转换，保留原始字段及变换记录；归一化数据先恢复尺度。每批数据先用不少于20个静态点核对重投影方向与尺度，记录中位像素误差，转换失败的数据不进入已知位姿验证。',indent=0)
replace(27,14,18,'对没有位姿的真实航拍图，使用几何估计方法构建相对点图；多参考图只从训练组同一场景中选取，先掩蔽人车等动态对象再估计背景相机。单图条件利用粗点云构造相对坐标系，以参考中心深度为归一化半径，半径变化只表示相对比例。几何估计失败、天空占比过高或静态重叠不足时，登记失败并分析适用范围。小目标深度可由邻域地面估计作为采样提示，其误差不作为物理真值。')
replace(27,19,20,'（3）场景生成与目标相关视角选择。场景生成以粗几何的相机渲染序列作为条件，结合图像与生成先验构造新的观测。概念上表示为',indent=17)
replace(27,22,26,'其中，Θ为冻结生成权重，C为相机序列，P为粗点云，R为点渲染，ξ为种子对应噪声，L为轨迹帧数。轨迹长度、图像分辨率、采样步数及数值精度由单卡显存与吞吐预实验确定，随后在各比较组保持一致，并保存全部随机种子。该式描述图像及几何先验下的条件生成，不意味着恢复未观测的真实内容。逐帧解码与分块处理用于降低显存需求，配置变化登记为独立实验。',indent=0)
replace(27,27,29,'在每个训练参考场景中，对俯仰角、方位角和相对半径分别设置三个离散水平，去掉无变化条件，形成26个候选终点。初始角度幅度分别为10°和15°，半径变化幅度为10%，只允许在开发集调整一次。采样采用统一球面轨迹约定，运行前核对方向并保存实际相机矩阵作为最')
replace(28,6,12,'候选需满足静态背景投影覆盖不低于参考有效像素的60%、相对半径变化不超过10%，再贪心选择3条轨迹。评分权重与覆盖阈值只在开发组比较后冻结，不针对最终测试改动。随机采样与均匀角度采样使用相同候选集合、轨迹数、生成步数及种子。主实验最多使用100个真实训练参考场景，每个场景最多3条轨迹。轨迹长度在资源预实验后固定，每条轨迹按相同相对时间位置选择4帧，至多1200张进入人工补标预算，其余帧仅用于轨迹验证。各组按实际完成的同源清单与标注数量作等预算比较。')
replace(28,26,30,'（5）下游训练与验证。检测基线选用公开预训练目标检测模型，固定权重来源、输入尺度、优化器、有效批量、学习率与增强配置。上述配置根据开发集和资源预实验确定，最终比较前冻结。训练使用相同优化步数及随机种子，统一几何与颜色增强。生成样本按真实样本数量比例0、0.25、0.5进行有放回采样，防止新增图像直接增加优化步',indent=17)
replace(29,18,23,'（1）单智能体构成与固定任务链。目标检测模块提供定位证据，视觉语言模型提供任务解析、受限工具选择和说明生成，规则引擎计算等级，经验库提供适用策略。主实验冻结语言基座及视觉编码器，仅对语言解码器中的轻量适配参数进行更新；检测模块使用研究点一确定并冻结的版本。检测模块继续训练作为独立扩展实验。所有模型检查点、分词与图像处理配置、提示模板均记录校验值，禁止实验中自动升级。',indent=17)
replace(30,3,8,'图像输入保留原始长宽信息，图像像素预算、文本上下文长度和输出长度通过显存预实验确定，并在各组固定。推理采用单候选确定性解码。提示模板要求先返回结构化调用或终止动作，再由报告模板引用证据，模型自由文本不得直接修改风险等级。初始源域适配使用训练图像构造的目标、工具与规则问答，统一有效批量、学习率及训练步数，以开发集结构化任务分数选取检查点，之后冻结为在线适应起点。')
replace(32,6,7,'（4）轻量参数更新与能力保持。在语言解码器的注意力投影中引入低秩参数增量：',indent=17)
replace(32,19,24,'参数更新采用带权重衰减的自适应优化，学习率、有效批量和梯度裁剪阈值在开发集确定后冻结，结合混合精度、梯度累积与梯度检查点控制显存。每次候选最多50个优化步骤，源域回放与当前代理样本按1:1交替。有效代理样本少于32条则跳过参数更新并登记原因。出现非有限损失、显存不足或20分钟训练预算耗尽时，终止候选并保留活动版本。保护集不参与梯度训练，候选仅在最终步骤执行一次保护检查。')
replace(32,25,32,'（5）经验库、更新周期与候选准入。在线流按原始时间顺序运行，每128个已提交任务形成一个更新窗口。先进先出的原始缓冲区最多512条，最多保留最近4个窗口；冻结源域回放池最多512条，按任务类型和类别分层从训练集选取，各组使用相同清单。经验库上限1000条，记录适用任务、可观测条件、动作摘要、执行结果、证据指针、规则与检测版本、创建窗口及有效期。近重复经验按规范化动作和错误类型的内容摘要去重；检索先按任务、错误类型和版本兼容性过滤，再进行固定词项匹配排序，取前3条，并列按创建编号排列。分词、检索权重和排序规则均在开发阶段冻结，避免不受控的检索模型更新。',indent=17)
replace(34,11,15,'（1）总体架构与运行流程。平台通过前端界面呈现数据、任务及风险证据，由控制服务提供输入校验与任务接口，结构化存储保存记录和任务队列，对象存储保存图像与模型等大文件，并通过隔离运行环境固定依赖。生成、推理及适应训练在独立工作进程中运行，单卡任务按设备租约串行调度。各模块按接口通信并分别保存环境配置，降低依赖兼容问题对复现的影响。架构如图10所示。',indent=17)
replace(36,2,6,'（4）工具接口、日志与报告。模块输入包括协议版本、运行编号、图像内容摘要、尺寸、模型版本及原图坐标定义；输出包括状态、对象编号、类别、分数、坐标、耗时和错误信息。规则接口只接受经过结构检查的检测对象与区域多边形，返回规则事件及等级。报告接口核对引用对象与区域存在后写入说明文本，并保存对应证据。',indent=17)
replace(40,7,10,'公开方法在复杂航拍场景中的小目标表现、补充风险标注的充分性以及轻量适配代理信号的有效性均需实际验证。研究设计保留均匀生成、固定规则链、仅记忆和无更新等可运行基线，使任一方法未获得收益时，仍可给出可解释的比较结论。')
replace(40,18,22,'公开合成城市场景与真实航拍数据为已知位姿生成验证及真实检测提供研究入口。风险任务需补充区域和障碍物，拟以最多3000帧真实标注和1200帧生成标注控制起始工作量。前两个月重点核查场景覆盖与类别支持数；样本不足时公开实际规模和评价限制，不以复制样本或任意修改类别扩大数据量。')
replace(40,27,29,'实验以单张24GB显存设备为起点，生成、检测训练与智能体适应顺序运行。通过预实验确定模型规模、输入尺度及数值精度，结合分块计算和梯度累积控制显存。各方法固定有效训练预算，并记录实际资源占用。')
replace(41,0,1,'先进行10条样本的显存与吞吐试运行。配置冻结后不自动降低分辨率；超出显存预算时，另设资源受限实验组并报告其与主设置的差异。',indent=0)
replace(25,20,25,'对合成城市场景，不以相邻视图随机划分训练和测试。先按相机水平位置构建地块，初始边长为100米，并依据视锥交叠及共享道路结构合并高度重叠地块。跨划分移除相邻边界视图。以固定随机种子按场景组分为60%开发训练、20%开发验证和20%最终生成测试，保留原始划分字段供对照。主实验最多选择60个训练组、20个开发组和20个测试组，每组最多3个参考配置；不足时使用全部合格组并公开实际数量。每个测试配置仅输入指定的1张')
replace(38,0,1,'（2）生成验证指标。已知相机的合成场景保留目标视图，用于测量像素误差与结构相似性。仅在静态有效区域计算：',indent=17)
replace(38,7,13,'其中，彩色图像统一缩放至单位区间，目标图未参与生成条件。结构相似性采用固定窗口及参数；感知距离使用冻结的特征提取网络，在静态有效的图像对上计算，不将动态掩码边缘的伪差异计入主比较。相机评价记录静态点重投影误差、候选生成成功率与拒绝原因；任务相关评价记录目标数量保持、重标后可见目标比例及尺寸分布。真实未知位姿图像没有配对真值，不报告伪造的像素指标；生成图外观或数量相似不能替代真实检测测试。',indent=0)

# Short generic replacements in unchanged lines or diagram labels.
subs={'ViewCrafter':'场景生成方法','DUSt3R':'几何估计','YOLOv8s':'检测模型','YOLOv8':'检测模型','Qwen2.5-VL-7B-Instruct':'视觉语言模型','LoRA':'低秩适配','MatrixCity':'合成城市场景','VisDrone2019-DET':'公开航拍数据','VisDrone DET':'公开航拍检测数据','VisDrone':'航拍数据','OpenCV':'通用','AlexNet':'固定感知特征','inspect_roi':'区域复检','48GB':'24GB'}
# Non-overlapping leftovers: redacted paragraphs and new diagrams are excluded.
figure_rects={25:(88,179,506,452),29:(88,295,506,482),31:(88,436,506,620)}
for n,r in figure_rects.items():erase(n,r,True)
# Rebuild architecture figure as well so every implementation label is neutral and aligned.
erase(34,(89,477,506,700),True)
erase(26,(73,94,523,235),True)
# Explicit generalization of grouped diagram/table labels.
line_replace(34,5,'参数候选\n语言层轻量适配\n源域回放与保持约束') if False else None
for n,blocks in original_blocks.items():
 if n in range(16,22) or n==42:continue # Preserve bibliographic records.
 for k,b in enumerate(blocks):
  r=fitz.Rect(b[:4]);t=b[4].strip()
  if re.match(r'图\d+\s*[:：]',t):continue
  if any(r.intersects(q[0]) for q in queue.get(n,[])):continue
  new=t
  if n<22:
   # Review names identify prior work, rather than prescribe implementation.
   new=new.replace('本课题使用VisDrone','本课题使用公开航拍数据').replace('将MatrixCity 用于','将已知相机的合成场景用于')
  else:
   for a,c in subs.items():new=new.replace(a,c)
  if new==t:continue
  if '\n' in t:
   # Preserve grouped table/diagram cells; replace each original line independently.
   for line in doc[n-1].get_text('dict')['blocks']:
    if line.get('type')!=0:continue
    for ln in line['lines']:
     lr=fitz.Rect(ln['bbox']);lt=''.join(s['text'] for s in ln['spans'])
     if not r.contains(lr):continue
     nt=lt
     for a,c in subs.items():nt=nt.replace(a,c)
     if nt==lt:continue
     erase(n,lr+(-.1,-.1,.1,.1))
     def cell(n=n,lr=lr,nt=nt,ln=ln):
      fs=min(ln['spans'][0]['size'], max(9.2,lr.width/max(width(nt,10),1)*10))
      put(doc[n-1],lr.x0,ln['spans'][0]['origin'][1],nt,fs)
     jobs.append(cell)
  else:line_replace(n,k,new)

# Remove appendices from the table of contents and retain section 7 as the endpoint.
erase(4,(73,205,532,485),True)
# Research schedule: a continuous one-year plan with complete prose around the table.
erase(42,(73,282,533,748),True)

def centered(page,rect,text,fs=10,style='sans',color=(.13,.20,.29),leading=None):
 r=fitz.Rect(rect);leading=leading or fs*1.38
 lines=text.split('\n')
 while max(width(s,fs,style) for s in lines)>r.width-8:fs-=.1;leading=fs*1.38
 y=r.y0+(r.height-leading*len(lines))/2+fs*.97
 for s in lines:
  put(page,(r.x0+r.x1-width(s,fs,style))/2,y,s,fs,style,color);y+=leading

def box(page,rect,title,body='',accent='blue',fs=9.5):
 r=fitz.Rect(rect)
 palette={'blue':((.94,.97,.995),(.43,.59,.73)), 'green':((.94,.98,.97),(.38,.62,.58)), 'grey':((.97,.975,.98),(.66,.71,.77)), 'amber':((1,.98,.92),(.79,.64,.37))}
 fill,border=palette[accent]
 page.draw_rect(r,color=border,fill=fill,width=.65)
 if body:
  centered(page,(r.x0+3,r.y0+3,r.x1-3,r.y0+19),title,fs+.2)
  centered(page,(r.x0+3,r.y0+21,r.x1-3,r.y1-2),body,fs-.3,color=(.23,.28,.34))
 else:centered(page,r,title,fs)

def arrow(page,points,label=None,label_pos=None):
 pts=[fitz.Point(p) for p in points];c=(.35,.43,.51)
 sh=page.new_shape();sh.draw_polyline(pts);sh.finish(color=c,width=.75,closePath=False);sh.commit()
 a,b=pts[-2],pts[-1];ang=math.atan2(b.y-a.y,b.x-a.x);size=3.5
 p1=fitz.Point(b.x-size*math.cos(ang-.45),b.y-size*math.sin(ang-.45));p2=fitz.Point(b.x-size*math.cos(ang+.45),b.y-size*math.sin(ang+.45))
 page.draw_polyline([p1,b,p2],color=c,width=.75)
 if label and label_pos:put(page,*label_pos,label,8,'sans',c)

def figure6():
 p=doc[24];xs=[90,232,374];w=132;centers=[156,298,440]
 box(p,(90,184,506,221),'道路与园区巡检任务','人员、车辆、障碍物检测与区域规则风险分级',fs=10)
 for c in centers:arrow(p,[(298,221),(298,231),(c,231),(c,241)] if c!=298 else [(c,221),(c,241)])
 titles=['研究内容一','研究内容二','研究内容三']
 bodies=['同场景新视角生成\n目标相关视角选择','参数适应与经验更新\n能力保持与候选检查','任务编排与证据管理\n版本回放与故障恢复']
 titles2=['数据与标注版本','模型与经验版本','运行清单与状态快照']
 bodies2=['生成来源与相机条件\n训练样本与标注清单','冻结检测模块与规则\n参数、记忆独立消融','证据、异常与重试\n可恢复任务记录']
 for x,c,t,b,t2,b2 in zip(xs,centers,titles,bodies,titles2,bodies2):
  box(p,(x,242,x+w,299),t,b,fs=9.7);arrow(p,[(c,299),(c,318)])
  box(p,(x,319,x+w,377),t2,b2,'green',9.5)
  arrow(p,[(c,377),(c,390),(298,390),(298,404)] if c!=298 else [(c,377),(c,404)])
 box(p,(90,405,506,449),'分层评价与结果分析','真实检测收益；风险分级与旧能力保持\n任务成功率、资源开销与回放一致性','amber',9.5)

def figure7():
 p=doc[28];xs=[90,232,374];w=132
 # Snake sequence removes ambiguous connections in the original drawing.
 data=[(0,301,'1  输入与来源冻结','单图或稀疏参考\n相机条件与场景编号'),(1,301,'2  粗几何与目标条件','相对几何估计\n静态背景与动态掩码'),(2,301,'3  候选视角评分','尺度、可见性与覆盖\n相机位移约束'),(2,367,'4  新视角生成','冻结生成模型\n固定轨迹长度与种子'),(1,367,'5  新视图独立标注','对象框与可见属性\n描述与有限问答'),(0,367,'6  固定预算检测训练','真实图与生成图\n二维增强及采样对照')]
 for col,y,title,body in data:box(p,(xs[col],y,xs[col]+w,y+53),title,body,'blue' if y==301 else 'green',9.3)
 arrow(p,[(222,327.5),(231,327.5)]);arrow(p,[(364,327.5),(373,327.5)])
 arrow(p,[(440,354),(440,366)])
 arrow(p,[(374,393.5),(365,393.5)]);arrow(p,[(232,393.5),(223,393.5)])
 arrow(p,[(156,420),(156,429)])
 box(p,(90,430,506,480),'两条独立验证线','已知位姿合成场景：相机与图像一致性；真实航拍：检测与扩增收益\n生成样本继承父场景划分，测试图像不参与训练扩增','amber',8.8)

def figure8():
 p=doc[30]
 # A left-to-right main route, explicit rejection branch, then a joint evidence report.
 box(p,(90,440,211,484),'输入与视觉证据','巡检图像、区域配置\n对象框、类别与置信度',fs=9.2)
 box(p,(235,440,356,484),'可判定性检查','区域有效性、目标可见性\n边界与阈值不确定带',fs=9.2)
 box(p,(380,440,506,484),'无法判断 U','记录证据缺失\n及无法判定原因','amber',9.2)
 arrow(p,[(211,462),(234,462)])
 arrow(p,[(356,462),(379,462)],'不足',(359,452))
 box(p,(90,509,211,552),'区域关系计算','足点、对象交叠\n区域覆盖代理','green',9.2)
 box(p,(235,509,356,552),'冻结规则映射','规则编号与触发条件\n输出风险等级 L0–L3','green',9.2)
 box(p,(380,509,506,552),'结构化结果汇合','保留逐区域等级与 U\n关联对象、区域和规则','green',9.2)
 arrow(p,[(295.5,484),(295.5,496),(150.5,496),(150.5,508)],'充分',(211,493))
 arrow(p,[(211,530.5),(234,530.5)]);arrow(p,[(356,530.5),(379,530.5)])
 arrow(p,[(443,484),(443,508)])
 arrow(p,[(443,552),(443,565)])
 box(p,(90,566,506,617),'统一证据报告','图像、对象、区域、规则与版本引用；风险等级与未知原因\n视觉语言模型组织说明，风险等级由规则引擎计算','grey',9.1)

def architecture():
 p=doc[33]
 box(p,(90,480,506,510),'前端交互：数据清单、任务配置、证据查看、版本比较与回放',fs=9.2)
 arrow(p,[(298,510),(298,521)])
 box(p,(90,522,506,551),'控制服务：输入校验、任务状态、版本解析、产物访问与报告',fs=9.2)
 for x in [156,298,440]:arrow(p,[(x,551),(x,562)])
 box(p,(90,563,222,618),'生成工作进程','独立环境与冻结模型\n生成图、相机、父样本','green',9.2)
 box(p,(232,563,364,618),'检测与智能体工作进程','定位与视觉语言推理\n区域规则与证据报告','green',9.2)
 box(p,(374,563,506,618),'适应工作进程','轻量参数与记忆更新\n候选检查与版本注册','green',9.2)
 box(p,(90,634,293,672),'结构化存储','任务、事件、版本依赖与经验索引','grey',9.2)
 box(p,(303,634,506,672),'对象存储','图像、权重、标注、报告与快照','grey',9.2)
 for x in [156,298,440]:arrow(p,[(x,618),(x,625),(191.5 if x<298 else 404.5,625),(191.5 if x<298 else 404.5,633)])
 centered(p,(90,678,506,699),'隔离环境固定依赖；单卡任务顺序运行；产物按内容摘要登记',8.7)

def data_table():
 p=doc[25];xs=[73.7,144,262,398,521.6];top=96;hh=20;rh=29
 p.draw_rect((xs[0],top,xs[-1],top+hh),fill=(.94,.96,.98),color=None)
 for j,t in enumerate(['数据来源','已有信息','拟补充或转换','用途与限制']):centered(p,(xs[j],top,xs[j+1],top+hh),t,9.5,'heading')
 rows=[('合成城市场景','图像、相机变换\n及可用深度','坐标约定转换；动态掩码；\n巡检目标区域','已知位姿新视角验证；\n独立于真实效果评价'),('真实航拍图像','对象框、类别\n及忽略信息','统一人员与车辆映射；\n障碍物完整增量标注','真实检测主验证；\n不假定位姿及风险真值'),('真实视频子集','序列帧及公开\n任务标注','区域、障碍物、规则事件\n及可判定性标注','真实风险及在线回放；\n按场景和序列隔离'),('本课题生成图','父图、相机条件\n及生成配置','新图独立对象框、可见性、\n描述和有限问答','仅用于训练；继承\n父场景划分及来源')]
 y=top+hh
 for i,row in enumerate(rows):
  if i%2==0:p.draw_rect((xs[0],y,xs[-1],y+rh),fill=(.985,.988,.99),color=None)
  for j,t in enumerate(row):centered(p,(xs[j]+1,y,xs[j+1]-1,y+rh),t,9.2,'body')
  y+=rh;p.draw_line((xs[0],y),(xs[-1],y),color=(.75,.79,.83),width=.35)
 for x in xs:p.draw_line((x,top),(x,y),color=(.68,.73,.79),width=.4)
 p.draw_line((xs[0],top),(xs[-1],top),color=(.35,.43,.52),width=.6)

def schedule():
 p=doc[41]
 paragraph(p,(73.7,289,521.6,334),'研究计划自通过开题后的第1个月起计算，共12个月，在一年内完成方法研究、实验验证、系统集成、论文撰写与答辩准备。具体阶段安排见表7。',12,24)
 centered(p,(73.7,342,521.6,364),'表7: 一年研究进度与阶段成果',11,'body')
 xs=[73.7,131.7,331.7,521.6];top=373;hh=25;rh=33
 p.draw_rect((xs[0],top,xs[-1],top+hh),fill=(.93,.95,.97),color=None)
 for j,t in enumerate(['时间','主要工作','阶段成果与检查']):centered(p,(xs[j],top,xs[j+1],top+hh),t,10,'heading')
 rows=[('第1—2月','文献与数据核查；场景划分、标注\n规范及检测基线','数据与标注清单；基线输出；\n确认任务和资源可行性'),('第3—4月','粗几何与生成基线；候选视角选择；\n图文样本重建与扩增对照','冻结生成预算；生成与标注版本；\n真实检测初步对照'),('第5—6月','单智能体、规则与证据链；\n参数适配和经验记忆基线','统一输出协议；完成参数、记忆\n各自的最小验证'),('第7—8月','连续适应、能力保持与候选恢复；\n错误经验与更新机制消融','新旧条件结果；更新开销；\n失败来源与适用边界'),('第9月','集成任务、版本和回放管理；\n实施故障注入与恢复验证','端到端原型；证据追溯报告；\n系统测试与恢复记录'),('第10月','冻结核心配置；完成多随机种子\n实验、独立评价与统计分析','测试结果归档；预算与泄漏核查；\n置信区间及误差分析'),('第11月','撰写论文；补充具有明确问题\n指向的实验与消融','论文初稿；方法局限分析；\n可复现材料初稿'),('第12月','论文与图表修订；答辩准备；\n代码、数据清单和实验归档','学位论文定稿；原型演示；\n完成答辩材料和研究归档')]
 y=top+hh
 for i,row in enumerate(rows):
  if i%2==0:p.draw_rect((xs[0],y,xs[-1],y+rh),fill=(.98,.985,.99),color=None)
  for j,t in enumerate(row):centered(p,(xs[j]+2,y,xs[j+1]-2,y+rh),t,9.5,'body')
  y+=rh;p.draw_line((xs[0],y),(xs[-1],y),color=(.78,.81,.85),width=.35)
 for x in xs:p.draw_line((x,top),(x,y),color=(.68,.73,.79),width=.45)
 p.draw_line((xs[0],top),(xs[-1],top),color=(.35,.43,.52),width=.7)
 p.draw_line((xs[0],top+hh),(xs[-1],top+hh),color=(.35,.43,.52),width=.55)
 paragraph(p,(73.7,678,521.6,748),'各阶段保留可运行基线与失败记录。第2个月完成数据和资源核查；样本不足时缩小实际实验规模并公开限制。第10个月前冻结最终评价配置，避免根据测试结果反复调参；生成扩增或在线更新未取得收益时，保留负结果及适用边界分析。',11.5,23,18.5)

# Apply all deletions first; preserve graphics except in explicitly rebuilt regions.
for n,items in queue.items():
 p=doc[n-1]
 # Redact diagrams/table paths separately: touching geometry is confined to these rectangles.
 for r,g in items:
  if g:p.add_redact_annot(r,fill=(1,1,1))
 if any(g for _,g in items):p.apply_redactions(images=0,graphics=1,text=0)
 for r,g in items:
  if not g:p.add_redact_annot(r,fill=(1,1,1))
 if any(not g for _,g in items):p.apply_redactions(images=0,graphics=0,text=0)
for job in jobs:job()
figure6();figure7();figure8();architecture();data_table();schedule()
# Remove the appendices and their bookmark entries.
toc=[t for t in doc.get_toc() if t[2]<=43 and '附录' not in t[1]]
doc.delete_pages(43,doc.page_count-1)
for link in doc[3].get_links():
 if fitz.Rect(link['from']).y0>205:doc[3].delete_link(link)
doc.set_toc(toc)
meta=doc.metadata;meta['title']='无人机巡检多模态智能体开题报告';meta['subject']='图题规范、流程图重排、一年研究进度与技术表述修订';doc.set_metadata(meta)
doc.subset_fonts()
result=OUT/'无人机巡检多模态智能体开题报告_排版修订版.pdf'
doc.save(result,garbage=4,deflate=True,clean=True)
(OUT/'revision_log.json').write_text(json.dumps({'missing_source_glyphs':sorted(missing),'revisions':log},ensure_ascii=False,indent=2))
print(result)
print('Pages',doc.page_count,'paragraphs',len(log),'fallback glyphs',''.join(sorted(missing)))
