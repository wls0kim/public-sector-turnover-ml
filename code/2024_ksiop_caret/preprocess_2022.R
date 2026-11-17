# 2022 공직생활실태조사 전처리 (01_classification.Rmd, 02_regression.Rmd에서 공통으로 사용)
#
# 1) MZ세대(1980년 이후 출생) 선택
# 2) 분석에 쓰지 않는 문항 제거 (복수응답, 하위 질문, 일부 기관만 응답한 문항, 가중치 등)
# 3) 결측 사례 제거
# 4) 역문항 처리
# 5) 다문항 요인은 문항 평균으로 바꾸고 원문항 제거
# 이직의도 원문항(q38)은 남겨 두고, 각 Rmd에서 분류용(yes/no) 또는 회귀용(1~5점)으로 바꾼다.

# 분석에서 제외하는 변수와 이유
drop_vars <- c(
  "ID",                                         # 응답자 ID
  "q1_1_m1", "q1_1_m2",                         # 업무량이 많은 이유 (복수응답)
  "q5_1_1", "q5_2_1", "q5_3_1",                 # 육아휴직, 보육시설, 유연근무제 만족도 (이용자만 응답)
  "q6_1_1", "q6_2_1", "q6_3_1", "q6_4_1",       # 재택근무, 스마트워크센터, 영상회의, 모바일 결재 만족도 (이용자만 응답)
  "A1", "A2",                                   # 통근거리, 통근시간 (광역, 기초자치단체만 응답)
  "q9_m1", "q9_m2", "q9_m3",                    # 승진에 영향을 미치는 요인 (복수응답)
  "q13_1_1", "q13_1_2", "q13_1_3",              # 복지제도 개선 항목
  "q17_m1", "q17_m2", "q18_m1", "q18_m2",       # 전문성 저해 요인, 강화 정책 (복수응답)
  "q28_m1", "q28_m2", "q28_m3",                 # 성과 향상 장애 요인 (복수응답)
  "B", "q31_C1", "q31_C2",                      # 비공식 관계, 현장업무, 악성민원 (광역, 기초자치단체만 응답)
  "q38_1", "q38_2_m1", "q38_2_m2",              # 이직 희망 분야, 이직 이유 (이직의도 하위 질문)
  "dq2", "dq7",                                 # 출생연도, 입직연도 (범주형 연령, 재직기간 변수로 대체)
  "DM2", "DM4", "DM6", "DM7",                   # 성별, 학력, 채용 유형, 직급 (설문 문항과 중복)
  "WT", "TWT"                                   # 가중치
)

reverse_items <- c("q8_5", "q14_4", "q19_5", "q33_5", "q33_6", "q33_7", "q33_8")

factor_items <- list(
  resource                            = paste0("q3_", 1:3),   # 자원제공
  job_autonomy                        = paste0("q4_", 1:3),   # 업무자율성
  recruitment                         = paste0("q7_", 1:4),   # 채용
  promotion                           = paste0("q8_", 1:5),   # 승진
  job_rotation                        = paste0("q10_", 1:3),  # 배치전환
  compensation_rewards                = paste0("q11_", 1:4),  # 보수와 보상
  welfare                             = paste0("q13_", 1:2),  # 후생복지
  training                            = paste0("q14_", 1:4),  # 교육훈련, 능력발전
  performance_capacity                = paste0("q15_", 1:3),  # 업무수행 역량
  organizational_goals                = paste0("q19_", 1:5),  # 조직목표
  leadership                          = paste0("q20_", 1:7),  # 리더십
  decision_making                     = paste0("q22_", 1:4),  # 의사결정
  change_management                   = paste0("q23_", 1:3),  # 변화관리
  collaboration_communication         = paste0("q24_", 1:3),  # 협업, 의사소통
  organizational_performance          = paste0("q25_", 1:2),  # 조직성과
  performance_management              = paste0("q26_", 1:5),  # 성과관리
  job_performance                     = paste0("q27_", 1:3),  # 업무성과
  public_service                      = paste0("q30_", 1:6),  # 공공봉사동기
  organizational_citizenship_behavior = paste0("q32_", 1:5),  # 조직시민행동
  innovative_behavior                 = paste0("q33_", 1:8),  # 혁신행동
  public_service_values               = paste0("q34_", 1:8),  # 공직가치
  commitment                          = paste0("q35_", 1:4),  # 조직몰입
  job_satisfaction                    = paste0("q36_", 1:3),  # 직무만족
  public_job_satisfaction             = paste0("q37_", 1:4),  # 공직만족도
  quality_of_life                     = paste0("q39_", 1:2)   # 삶의 질
)

preprocess_2022 <- function(raw) {
  raw <- raw[raw$dq2 >= 1980, ]                    # MZ세대
  raw <- raw[, !(names(raw) %in% drop_vars)]
  data <- na.omit(raw)
  cat("결측 제거 전후 사례 수:", nrow(raw), "→", nrow(data), "\n")

  data[reverse_items] <- lapply(data[reverse_items], function(x) 6 - x)

  for (f in names(factor_items)) {
    data[[f]] <- rowMeans(data[, factor_items[[f]]])
  }
  data[, !(names(data) %in% unlist(factor_items))]
}
