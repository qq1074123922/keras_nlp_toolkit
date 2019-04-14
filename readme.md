# NLP 工具箱

这里汇聚各个 NLP 工具，预料处理器等。

## 安装方法
```
git pull ssh://root@github.com/AILab-aida/nlp_toolkit.git
cd nlp_toolkit
pip install ./

# 如果需要编辑代码
pip install -editable ./  
```

使用方法

```
import nlp_toolkit
s = nlp_toolkit.SentenceCutter()

---------
from nlp_toolkit import SentenceCutter
s = SentenceCutter()
```


##警告
keras API已经更新，本项目不在维护。