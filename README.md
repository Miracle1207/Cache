# Cache
此处与【cache+BLAv1.0】相比，
1. 改进了 BLA中reward与penalty的设置，feedback改为了"Delay(t-1)-Delay(t)"，大于0为reward，小于0为penalty。
每过25个step，feedback = Delay(t-25)-Delay(t).
2. 加入Decimal()解决了计算probability of action中阶乘计算溢出的问题。
