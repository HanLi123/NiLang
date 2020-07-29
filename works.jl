using Plots
using NiLang
using NiLang.AD: Grad, grad, check_grad
using Statistics: std, mean

using DelimitedFiles

#提取日线沪深300指数收盘价
b = vec(readdlm(joinpath(@__DIR__, "data.txt")))

#EMA权重
bili=0.0952
n=length(b)
ma=zeros(n,1)
ma[1]=b[1]
pos=zeros(n,1)
#单位净值
jinzhi=ones(n,1)
function regular_jin(b)
    for i =2:n
        if pos[i-1]>0
            #如果信号为正,做多
            jinzhi[i]=jinzhi[i-1]*(b[i]/b[i-1]);
        elseif pos[i-1]<0
            #如果信号为负,做空
            jinzhi[i]=jinzhi[i-1]*(2-b[i]/b[i-1]);
        else
            #如果信号为0,保持不变
            jinzhi[i]=jinzhi[i-1];
        end
        #计算EMA
        ma[i]=b[i]*bili+(1-bili)*ma[i-1];

        if ma[i]<b[i]
            #如果上穿均线，信号为正
            pos[i]=1;
        else
            #如果下穿均线，信号为负
            pos[i]=-1;
        end
    end
    #计算夏普率作为loss
    mean(jinzhi)/std(jinzhi), jinzhi
end

shapu, jinzhi = regular_jin(b)
plot(jinzhi)

@i function jin(out::T,b::AbstractVector{T},ma::AbstractVector{T},jinzhi::AbstractVector{T},pos::AbstractVector{T},bili::T) where T
    @routine begin
        for i =2:length(b)
            jinzhi[i] += jinzhi[i-1]
            if (pos[i-1]!=0, ~)
                @routine begin
                    @zeros T anc1 anc2
                    anc1 += b[i] / b[i-1]
                    anc1 -= 1
                    anc2 += anc1 * pos[i-1]
                    anc2 += 1
                end
                jinzhi[i] -= jinzhi[i-1]
                jinzhi[i] += jinzhi[i-1] * anc2
                ~@routine
            end

            ma[i] += b[i] * bili
            bili -= 1
            ma[i] -= bili * ma[i-1]
            bili += 1

            if (ma[i]<b[i], ~)
                pos[i] += 1.0
            else
                pos[i] -= 1.0
            end
        end
        @zeros T var varsum mean sum std
        NiLang.i_var_mean_sum(var, varsum, mean, sum, jinzhi)
        std += sqrt(var)
    end
    #计算夏普率
    out += mean / std
    ~@routine
end

bili=0.0952
n=length(b)
ma=zeros(n)
ma[1]=b[1]
jinzhi=zeros(n)
jinzhi[1]=1.0
pos=zeros(n)
out=0.0
@assert check_inv(jin, (out,b,ma,jinzhi,pos,bili))
@assert check_grad(jin, (out,b,ma,jinzhi,pos,bili); iloss=1)

# should be: 1.5511167937559058
out,b,ma,jinzhi,pos,bili=jin(out,b,ma,jinzhi,pos,bili)
out,b,ma,jinzhi,pos,bili=(~jin)(out,b,ma,jinzhi,pos,bili)
gout,gb,gma,gjinzhi,gpos,gbili= NiLang.AD.gradient(jin, (out,b,ma,jinzhi,pos,bili); iloss=1)
