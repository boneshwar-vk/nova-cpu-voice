question     run  1/5  ttfa   1412 ms  commit=soft p=0.93 chunks=2  dead_air=0  'What is a black hole in simple words?'
question     run  2/5  ttfa   1116 ms  commit=soft p=0.93 chunks=3  dead_air=0  'What is a black hole in simple words?'
question     run  3/5  ttfa   1088 ms  commit=soft p=0.93 chunks=3  dead_air=0  'What is a black hole in simple words?'
question     run  4/5  ttfa   1221 ms  commit=soft p=0.93 chunks=3  dead_air=0  'What is a black hole in simple words?'
question     run  5/5  ttfa   1114 ms  commit=soft p=0.93 chunks=3  dead_air=0  'What is a black hole in simple words?'
moon         run  1/5  ttfa   1489 ms  commit=soft p=0.99 chunks=2  dead_air=0  'How far away is the moon?'
moon         run  2/5  ttfa   1175 ms  commit=soft p=0.99 chunks=2  dead_air=0  'How far away is the moon?'
moon         run  3/5  ttfa   1248 ms  commit=soft p=0.99 chunks=2  dead_air=0  'How far away is the moon?'
moon         run  4/5  ttfa   1259 ms  commit=soft p=0.99 chunks=2  dead_air=0  'How far away is the moon?'
moon         run  5/5  ttfa   2088 ms  commit=soft p=0.99 chunks=2  dead_air=0  'How far away is the moon?'
roman        run  1/5  ttfa   1865 ms  commit=check p=0.76 chunks=2  dead_air=0  'Tell me something interesting about the Roman Empire.'
roman        run  2/5  ttfa   2600 ms  commit=check p=0.76 chunks=3  dead_air=0  'Tell me something interesting about the Roman Empire.'
roman        run  3/5  ttfa   2383 ms  commit=check p=0.76 chunks=3  dead_air=0  'Tell me something interesting about the Roman Empire.'
roman        run  4/5  ttfa   2421 ms  commit=check p=0.76 chunks=2  dead_air=0  'Tell me something interesting about the Roman Empire.'
roman        run  5/5  ttfa   1480 ms  commit=check p=0.76 chunks=3  dead_air=0  'Tell me something interesting about the Roman Empire.'
gravity      run  1/5  ttfa   1598 ms  commit=check p=0.81 chunks=3  dead_air=0  'Can you explain gravity again?'
gravity      run  2/5  ttfa   1593 ms  commit=check p=0.81 chunks=3  dead_air=0  'Can you explain gravity again?'
gravity      run  3/5  ttfa   1557 ms  commit=check p=0.81 chunks=3  dead_air=0  'Can you explain gravity again?'
gravity      run  4/5  ttfa   1830 ms  commit=check p=0.81 chunks=2  dead_air=0  'Can you explain gravity again?'
gravity      run  5/5  ttfa   1949 ms  commit=check p=0.81 chunks=2  dead_air=0  'Can you explain gravity again?'
incomplete   run  1/5  ttfa   1997 ms  commit=hard cap chunks=2  dead_air=0  'So I was thinking about the'
incomplete   run  2/5  ttfa   1980 ms  commit=hard cap chunks=2  dead_air=0  'So I was thinking about the'
incomplete   run  3/5  ttfa   2463 ms  commit=hard cap chunks=2  dead_air=0  'So I was thinking about the'
incomplete   run  4/5  ttfa   2131 ms  commit=hard cap chunks=2  dead_air=0  'So I was thinking about the'
incomplete   run  5/5  ttfa   2182 ms  commit=hard cap chunks=2  dead_air=0  'So I was thinking about the'

25 runs appended to bench/results/e2e_m2-8gb.jsonl

| clip | kind | n | ttfa p50 | p90 | min | max | timeouts | errors | dead-air | smart_turn_ms p50 | utter_ms p50 | stt_ms p50 | tts_first_ms p50 | memory_ms p50 | hangover_ms p50 | soft_after_speech_ms p50 | stt_after_commit_ms p50 | llm_ttft_ms p50 | llm_first_sentence_ms p50 | commit_to_audio_ms p50 | ttfa_ms p50 | reply_total_ms p50 | utt_audio_ms p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| question | complete | 5 | 1116 | 1336 | 1088 | 1412 | 0 | 0 | 0 | 123 | 2813 | 157 | 154 | 0 | 364 | 239 | 50 | 153 | 634 | 844 | 1172 | 1456 | 2813 |
| moon | complete | 5 | 1259 | 1849 | 1175 | 2088 | 0 | 0 | 0 | 125 | 1782 | 147 | 164 | 0 | 334 | 207 | 9 | 184 | 703 | 882 | 1216 | 1379 | 1782 |
| roman | complete | 5 | 2383 | 2529 | 1480 | 2600 | 0 | 0 | 0 | 193 | 3932 | 592 | 196 | 0 | 786 | 201 | 7 | 825 | 1304 | 1498 | 2300 | 1906 | 3932 |
| gravity | complete | 5 | 1598 | 1901 | 1557 | 1949 | 0 | 0 | 0 | 155 | 2980 | 591 | 135 | 0 | 795 | 205 | 4 | 113 | 634 | 760 | 1518 | 1497 | 2980 |
| incomplete | incomplete | 5 | 2131 | 2350 | 1980 | 2463 | 0 | 0 | 0 | 198 | 3245 | 1387 | 77 | 0 | 1590 | 203 | 1 | 90 | 380 | 488 | 2059 | 1048 | 3245 |
| **overall** | mixed | 25 | 1598 | 2405 | 1088 | 2600 | 0 | 0 | 0 | 153 | 2980 | 563 | 145 | 0 | 764 | 205 | 6 | 153 | 685 | 844 | 1518 | 1456 | 2980 |
