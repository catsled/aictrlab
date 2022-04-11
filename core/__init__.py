"""
这里预定义一些东西，以处理不同场景下的问题。

包含关系：
Agent
  -- policy
    -- network
    -- optimizer
      -- lr
  -- update
    -- update_epoch
  -- other strategies
    -- explore rate decay

网络输入规则：Agent并不负责状态的处理，其只需要知道选取何种网络，网络对应的参数。

图像：统一使用opencv对图像的处理方式，该步骤同样在Agent外部完成，图像的长高必须为偶数，输入顺序符合pytorch规范。

异常处理：在Agent内部不进行任何输入的检查，内部产生的异常则正常处理。
"""