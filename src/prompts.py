from __future__ import annotations

from .config import SENTIMENT_LABELS, TOPIC_LABELS


def sample_few_shot_examples(train_df, n_per_label=1, seed=42):
    examples = []
    sent_df = train_df.groupby("sentiment", group_keys=False).apply(
        lambda x: x.sample(min(n_per_label, len(x)), random_state=seed)
    )
    topic_df = train_df.groupby("topic", group_keys=False).apply(
        lambda x: x.sample(min(n_per_label, len(x)), random_state=seed)
    )
    examples.append(("sentiment", sent_df[["sentence", "sentiment"]].drop_duplicates().to_dict("records")))
    examples.append(("topic", topic_df[["sentence", "topic"]].drop_duplicates().to_dict("records")))
    return examples


def build_joint_prompt(sentence: str, few_shot_rows=None):
    intro = '''Bạn là chuyên gia gán nhãn phản hồi sinh viên bằng tiếng Việt.

Nhiệm vụ:
1) sentiment: một trong ["negative", "neutral", "positive"]
2) topic: một trong ["lecturer", "curriculum", "facility", "others"]

Định nghĩa:
- negative: phàn nàn, chê, không hài lòng
- neutral: mô tả trung tính, góp ý không rõ khen/chê
- positive: khen, hài lòng, đánh giá tốt
- lecturer: nói về giảng viên, cách giảng dạy, thái độ, khả năng truyền đạt
- curriculum: nói về chương trình học, nội dung môn, lịch học, tải học tập
- facility: nói về cơ sở vật chất, phòng học, thiết bị
- others: không thuộc ba nhóm trên
'''
    shots = ""
    if few_shot_rows:
        shots += "\nVí dụ:\n"
        for row in few_shot_rows:
            shots += (
                f'Câu: "{row["sentence"]}"\n'
                f'JSON: {{"sentiment": "{row["sentiment"]}", "topic": "{row["topic"]}"}}\n\n'
            )
    ending = f'''Chỉ trả về JSON hợp lệ đúng schema:
{{"sentiment":"...", "topic":"..."}}

Câu phản hồi:
"{sentence}"'''
    return intro + shots + ending