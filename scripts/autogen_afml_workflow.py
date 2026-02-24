import os
import asyncio
from typing import Annotated

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# --- 1. afmlkit Mock Tools ---

def generate_dollar_bars_tool(raw_data_path: Annotated[str, "原始数据所在的存储路径"], target_freq: Annotated[int, "目标生成频率参数"]) -> str:
    """根据原始交易数据生成动态 Dollar Bars，以恢复数据的正态性并同步信息流。"""
    print(f"🔧 [Tool Call] generate_dollar_bars_tool(raw_data_path='{raw_data_path}', target_freq={target_freq})")
    try:
        import pandas as pd
        from afmlkit.bar.data_model import TradesData
        from afmlkit.bar.kit import DynamicDollarBarKit

        import tables
        import datetime
        from dateutil.relativedelta import relativedelta
        import gc

        print(f"Loading 1 year of recent data from {raw_data_path}...")
        
        # Calculate trailing 12 months keys
        end_date = datetime.date(2026, 1, 1) # Latest data month from filename '2601'
        months_to_load = [(end_date - relativedelta(months=i)).strftime('%Y-%m') for i in range(12)]
        months_to_load.reverse()

        import numpy as np
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False

        ts_list, px_list, qty_list = [], [], []
        
        for ym in months_to_load:
            if has_psutil:
                mem = psutil.virtual_memory()
                # 如果可用内存低于 2GB，则停止加载更早的历史数据以防止 OOM 崩溃
                if mem.available < 2 * 1024**3:
                    print(f"⚠️ Warning: Available memory ({mem.available/1024**3:.2f} GB) is too low. Stopping data loading at {ym}.")
                    break

            key = f'/trades/{ym}'
            try:
                # 仅读取构建 Dollar Bars 所需核心列，避免加载其它无关字段，极大减少内存占用
                month_df = pd.read_hdf(raw_data_path, key=key, columns=['timestamp', 'price', 'amount'])
                ts_list.append(month_df['timestamp'].values)
                px_list.append(month_df['price'].values)
                qty_list.append(month_df['amount'].values)
                
                if has_psutil:
                    print(f"Loaded {key} with {len(month_df)} rows. Avail Mem: {psutil.virtual_memory().available/1024**3:.2f} GB")
                else:
                    print(f"Loaded {key} with {len(month_df)} rows")
                
                del month_df
                gc.collect()
            except KeyError:
                print(f"Skipping {key} because it does not exist.")
        
        if not ts_list:
            return "Error: No trades data found for the past 12 months."
            
        # Initialize TradesData直接拼接 numpy 数组，避免生成庞大的临时 Pandas DataFrame
        trades = TradesData(
            ts=np.concatenate(ts_list),
            px=np.concatenate(px_list),
            qty=np.concatenate(qty_list),
            timestamp_unit='ns'
        )
        
        del ts_list, px_list, qty_list
        gc.collect()

        db_kit = DynamicDollarBarKit(trades, target_daily_bars=target_freq)
        bars_df = db_kit.process()

        output_path = f"{raw_data_path.split('/')[-1].split('.')[0]}_dollarbars.parquet"
        bars_df.to_parquet(output_path)
        
        return f"Dollar bars successfully generated! Target freq: {target_freq}. Result saved to {output_path} with {len(bars_df)} bars."
    except Exception as e:
        return f"Error generating dollar bars: {e}"

def adf_test_tool(data_path: Annotated[str, "数据路径"]) -> str:
    """执行 Augmented Dickey-Fuller (ADF) 检验，检查数据平稳性。"""
    print(f"🔧 [Tool Call] adf_test_tool(data_path='{data_path}')")
    return "ADF Test Result: p-value = 0.15 (Non-stationary)."

def apply_frac_diff_tool(data_path: Annotated[str, "需要平稳化的数据"], d: Annotated[float, "分数值（例如0.45）"]) -> str:
    """使用分数阶差分 (FracDiff) 保留记忆的同时让数据平稳。绝不使用整数差分(d=1)。"""
    print(f"🔧 [Tool Call] apply_frac_diff_tool(data_path='{data_path}', d={d})")
    output_path = f"{data_path}_fracdiff.parquet"
    return f"FracDiff mapping constructed with d={d}. Data preserved memory and is now stationary. Saved to {output_path}."

def apply_triple_barrier_tool(data_path: Annotated[str, "平稳的价格特征数据"]) -> str:
    """应用三重屏障法 (Triple-Barrier Method) 进行科学打标。第一触碰决定标签。"""
    print(f"🔧 [Tool Call] apply_triple_barrier_tool(data_path='{data_path}')")
    return f"Triple barriers applied. Target labels created on {data_path}."

def calculate_sample_weights_tool(labels_path: Annotated[str, "打标后的数据"]) -> str:
    """计算并发重叠样本的平均唯一性 (Average Uniqueness) 进行样本降权。"""
    print(f"🔧 [Tool Call] calculate_sample_weights_tool(labels_path='{labels_path}')")
    return f"Sample weights calculated based on uniqueness for {labels_path}. Ready for modeling."

def purged_kfold_cv_tool(features_labels_path: Annotated[str, "特征和标签数据路径"]) -> str:
    """严格执行净化 K 折交叉验证 (Purged K-Fold CV)，加入 Embargo 阶段防止信息泄露。"""
    print(f"🔧 [Tool Call] purged_kfold_cv_tool(features_labels_path='{features_labels_path}')")
    return "Purged K-Fold CV completed, avoiding overlapping leakage."

def mda_feature_importance_tool(model_path: Annotated[str, "已训练模型路径"]) -> str:
    """使用 MDA/MDI 剔除冗余特征。"""
    print(f"🔧 [Tool Call] mda_feature_importance_tool(model_path='{model_path}')")
    return "MDA evaluated. Irrelevant features dropped."

def meta_labeling_sizing_tool(model_path: Annotated[str, "主模型路径"]) -> str:
    """通过元标签 (Meta-Labeling) 建立第二层模型，以决定头寸大小。"""
    print(f"🔧 [Tool Call] meta_labeling_sizing_tool(model_path='{model_path}')")
    return "Meta-labeling finished. Position sizing logic based on probability implemented."

def calculate_dsr_tool(returns_path: Annotated[str, "策略历史回测收益率路径"], num_trials: Annotated[int, "测试过的策略特征组合次数"]) -> str:
    """计算缩水夏普比率 (Deflated Sharpe Ratio, DSR) 以避免多重测试偏差。决定最终策略是否通过。"""
    print(f"🔧 [Tool Call] calculate_dsr_tool(returns_path='{returns_path}', num_trials={num_trials})")
    prob = 0.96 
    return f"DSR Probability: {prob}. Status: {'PASSED' if prob > 0.95 else 'FAILED'}"

# --- 2. AutoGen Registration ---
tool_gen_bars = FunctionTool(generate_dollar_bars_tool, description="生成 Dollar Bars 代替时间K线")
tool_adf_test = FunctionTool(adf_test_tool, description="执行 ADF 平稳性检验")
tool_frac_diff = FunctionTool(apply_frac_diff_tool, description="应用分数阶差分让时间序列平稳")
tool_triple_barrier = FunctionTool(apply_triple_barrier_tool, description="应用三重屏障法进行科学打标")
tool_sample_weights = FunctionTool(calculate_sample_weights_tool, description="计算样本唯一性权重以避免重叠偏差")
tool_purged_cv = FunctionTool(purged_kfold_cv_tool, description="执行带 Embargo 的 Purged K-Fold 组群训练")
tool_mda = FunctionTool(mda_feature_importance_tool, description="使用 MDA 计算特权重要性")
tool_meta_label = FunctionTool(meta_labeling_sizing_tool, description="执行元标签计算头寸大小")
tool_calc_dsr = FunctionTool(calculate_dsr_tool, description="使用 DSR 验证策略以避免发现虚假策略")

# --- 3. Multi-Agent Orchestration ---
async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Warning: OPENAI_API_KEY not found in environment variables. Using a mock key might fail real execution.")
        api_key = "sk-mock-key-for-testing"
        
    model_client = OpenAIChatCompletionClient(
        model="gemini-3.1-pro-high",
        api_key="sk-jieyujing",
        base_url="http://192.168.100.99:8317/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown"
        }
    )
    
    # Initialize Agents
    data_engineer = AssistantAgent(
        name="data_engineer",
        model_client=model_client,
        tools=[tool_gen_bars, tool_adf_test, tool_frac_diff],
        handoffs=["labeler"],
        system_message="""你是首要的数据工程师。
[核心原则] 绝对不使用 Time Bars。必须将传入的 raw_data_path 转换为 Dollar Bars。
其次，你必须对生成的 Dollar Bars 执行 ADF 检验 (adf_test_tool)。如果不平稳，你必须使用 FracDiff (apply_frac_diff_tool)。
完成后，将平稳的特征数据路径通过转交给 labeler 代理进行下一环节的打标。"""
    )
    
    labeler = AssistantAgent(
        name="labeler",
        model_client=model_client,
        tools=[tool_triple_barrier, tool_sample_weights],
        handoffs=["quant_modeler"],
        system_message="""你是标注分析师。
承接 Data Engineer 传递过来的平稳数据路径。
[核心步骤]
1. 应用三重屏障法进行科学打标 (apply_triple_barrier_tool)。
2. 为了解决金融数据存在重叠(Overlapping)带来的过拟合问题，必须使用样本唯一性计算权重 (calculate_sample_weights_tool)。
完成后，通过转接工具交接给 quant_modeler 代理。"""
    )
    
    quant_modeler = AssistantAgent(
        name="quant_modeler",
        model_client=model_client,
        tools=[tool_purged_cv, tool_mda],
        handoffs=["strategist"],
        system_message="""你是量化建模师。
承接 Labeler 代理 的打标结果。
[核心步骤]
1. 执行带有 Embargo 的 Purged K-Fold CV 防止信息泄露。
2. 使用 MDA 评价并剔除冗余特征。
完成后，通过转接工具交接给 strategist 代理。"""
    )
    
    strategist = AssistantAgent(
        name="strategist",
        model_client=model_client,
        tools=[tool_meta_label, tool_calc_dsr],
        system_message="""你是策略验证官，AFML 体系的最后防线。
承接 Quant Modeler 的输出结果。
[核心步骤]
1. 使用元标签确定策略下注大小 (meta_labeling_sizing_tool)。
2. 关键一步：对跑出的模型执行 DSR 验证计算 (calculate_dsr_tool, 设 num_trials=50)。
**只有当返回的 DSR 概率 > 0.95 时**，才算策略成功，此时必须在回复的最末尾输出字符串 'STRATEGY_APPROVED' 以终止项目。如果小于等于0.95，则报告失败。"""
    )
    
    termination = TextMentionTermination("STRATEGY_APPROVED")
    
    team = Swarm(
        participants=[data_engineer, labeler, quant_modeler, strategist],
        termination_condition=termination,
    )
    
    print("🚀 [System] Starting AFML Multi-Agent Workflow\\n" + "="*50)
    task_desc = "任务：基于 'data/h5/BTCUSDT_PERPum_2301-2601.h5' 设计全新的量化交易策略。请按照 AFML 标准管道执行数据工程、打标降权、建模并用 DSR 验证策略。"
    
    try:
        result = await team.run(task=task_desc)
        for message in result.messages:
            source = message.source if hasattr(message, 'source') else "System"
            content = str(message.content)
            # truncate long tool call JSON logs for readability
            # if len(content) > 300:
            #     content = content[:300] + "\n...[truncated]"
            # print(f"\\n👤 [{source}]:\\n{content}\\n---")
    except Exception as e:
        print(f"\\n❌ Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
