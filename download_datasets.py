#!/usr/bin/env python3
"""ECBRS 데이터셋 다운로드 스크립트"""

import os
import subprocess

# ids17 파일 ID 목록
ids17_files = {
    "9.npy": "1mG243VhCerbF3N7uOf5WYqdSLq-Vp_LJ",
    "10.npy": "1JD5pUrwEbk0WUWdyPSlTPCpsOSikhvx3",
    "11.npy": "1FCKsy8n-W8PdJOYYyszOXRNKwJkZgXCg",
    "12.npy": "1adqhWxcYt30860egzs4PoU_PdGZqtSru",
    "13.npy": "1OWQyE_V6XGkaE_2lXYDa0QgWNDTWbful",
    "14.npy": "1_YMenoVcHh6KpH_WIUJEP60xH_KMheSk",
    "15.npy": "1VWUvKWuELbshfdiNhJSWKfSP_BiozM0G",
    "16.npy": "1sMFhTqFUQyUyXz0TqkG1qLXmI-zOV-c3",
    "17.npy": "1gaiBHiyyMA9wmLBsFy3QwqqxyBsk21S_",
    "18.npy": "1ASO5sa4aR2J192yrr-OGRJgEtUt7YIT-",
    "19.npy": "1C-wDnLSdX14JIrS4BC1ojwoV384sAoiK",
    "20.npy": "1FaTsKOAsZ2FLMAFPsfOT-0sJxVaHx8_P",
    "21.npy": "17fpjYfiRx1jM7IuNKdh66lKw-m83OgKv",
    "22.npy": "1wmjNhI_RteF8-wGwZK9ISRBfrILq8_br",
    "23.npy": "1OThPYACXas0go19YQwnmxLVYjE1EZVyT",
    "24.npy": "1PUP5jRfP3LGjWMgo6z0qlTDe3GiCCtK9",
    "25.npy": "1oe00RFPWbsnswD_jUjkhb2cjbHdZyRgn",
    "26.npy": "1Q-Vf4r7gDUokywbJ13J2lN600u2y-wTg",
    # 테스트 파일들 (90-100)
    "90.npy": "1Se_QiZ0Lo4dLur4bm8Ytfku3XojXsEZT",
    "91.npy": "1bVDpxlZzvclyxaRnkZKz5ukUWk5jubq-",
    "92.npy": "1aj-zwC2_CuUGKebcPIYb7BYJHbBfrO4j",
    "93.npy": "1i05Q1Zohm4BPLSFyLfZzyd4TMSmcVYfA",
    "94.npy": "1aAK3-k8jfyVMq6sc4SkRy06dN8Ld32cY",
    "95.npy": "1HAp8CWj3metCXT8hOfSv3_GHuglWPimh",
    "96.npy": "1DhgGfpDGaHYX6NfhOScmtKs0roim_U0J",
    "97.npy": "13Oi9p6ZR3wN3uapJu56pPrHMlxMDNiaG",
    "98.npy": "1aPJ-DB9UfenWH-To--XYcmf3HegIR-QY",
    "99.npy": "1_8AUTYU-fV1kg2zi2O23xPa4N30DqP16",
    "100.npy": "1A157EjyGQxQw__TytRLYa85v21BSBY33",
}

# ids18 파일 ID 목록
ids18_files = {
    "1.npy": "1o-J95QrXpsIfhrMW80LLxA8vGBHSHiN3",
    "2.npy": "1uKtmHYHqlxxlHytqMfVTLqmgo7u15FyC",
    "3.npy": "1eAA5F5wzmrOj0yZes2GF-kkykfbcOMJB",
    "4.npy": "1rbhTVo8pb92kkC82y1WdVCqYe-FQHuVE",
    "5.npy": "15HowdLjfjlmzb_YsJT0se4j5CiT_sI0i",
    "6.npy": "15ZStyagOX8dPtFRF_7_Nd_qCclPWpEqa",
    "7.npy": "1cMwlf5d0X8SZydCV0svay8wUKgdZ9wEv",
    "8.npy": "1y4Xw044JPakqgKTBmXI_LHhOnxCiPMhB",
    "9.npy": "1flZwEetpPGel8pWNyy6tPMcLMZ7VUGQn",
    "10.npy": "19KJJqfQdWzkNQOqi99CbnvPIMLk9lvU8",
    "11.npy": "1DzaQK5vCyq7BufGv2JsgFm0Y5OHCYSYg",
    "12.npy": "1i8J7DycoAGVMAWXPlAP0KT0BmBoPA42Q",
    "13.npy": "1Hm2t2hUiCiBwATGeJQelOH4-9AWJa-Sb",
    "14.npy": "1qh8wB0Z4A3GmaZDZCCllSl9LyPzOWy-P",
    "15.npy": "1CDApyxF2p55QZL5DZMHY6SNZf-7J-sv-",
    "16.npy": "1qrv-yAyYd8IJEBjyGadtUJGZQCe0cxg9",
    "17.npy": "1x3dDc4IQ2m4VqQGqovCShYBrBKv_8Z4Y",
    "18.npy": "11cadYyUfsbMxMg3IEh2WEaY4CdN-6Z9D",
    "19.npy": "1rS07hU9yc7tnF_DzV4YQrllUbCa7afi-",
    "20.npy": "1RVpNS8-tOfAmNBHTj4sw8i1YqTE6lAli",
    "21.npy": "1Tp-xDvwpaNRT1koJ0mrAF0wPHoHIN7Lv",
    "22.npy": "1ijcGa14SEoR9OEHHySZ-Ftxv5Ep54K2t",
    "23.npy": "16_XmuaW2zDXv-BIO8acAv-201toiZKzg",
    "24.npy": "1RMJBa5Pm9mia7MU28tAKph_q5hDM1xTl",
    # 테스트 파일들
    "90.npy": "1Fvjw2PnbhmurwzzADfWbw6MDTTc1V2PU",
    "91.npy": "1rALrabhKBqV6vZk1I54KIm0XMlQIIHug",
    "92.npy": "1QHyXqA88c02mQZTomIrX81KhSE8uVrh7",
    "93.npy": "1r57olywS6guS6OJz5VHP961d5e-oidRq",
    "94.npy": "14Ux7xXLH7SDq0KSzWfizENW1xg3z1F_M",
    "95.npy": "1N6XkRQJr0nXF5NbYqswQn543lg3Q5hzl",
    "96.npy": "1sexMG4l-6WZPtV1sv-leygtDY9kPIJHF",
    "97.npy": "1jNHGBVRbhIn59atC-NwkX6MDl2E3qOiU",
    "98.npy": "1p0_MdakQKFD559bKuJQqaz0__RpV9xwe",
    "99.npy": "1XSJC1TTp-10LjgWeTaBZlMPooZhjcHCI",
    "100.npy": "1_qy42JeX4sA96QEBOxQJrjgu542CXQBg",
}


def download_file(fid, output):
    """curl로 Google Drive 파일 다운로드"""
    url = f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"
    cmd = ["curl", "-sL", "-o", output, url]
    subprocess.run(cmd, check=True)
    return os.path.getsize(output)


def download_dataset(name, files, base_dir="datasets/NEURIPS DATASET"):
    """데이터셋 다운로드"""
    dir_path = os.path.join(base_dir, name)
    os.makedirs(dir_path, exist_ok=True)

    downloaded = 0
    total_size = 0

    for fname, fid in sorted(files.items(), key=lambda x: int(x[0].split(".")[0])):
        output = os.path.join(dir_path, fname)
        if not os.path.exists(output) or os.path.getsize(output) < 1000:
            try:
                size = download_file(fid, output)
                print(f"  {fname}: {size:,} bytes")
                downloaded += 1
                total_size += size
            except Exception as e:
                print(f"  {fname}: FAILED - {e}")
        else:
            size = os.path.getsize(output)
            total_size += size

    print(
        f"\n  => {name}: {downloaded} 파일 다운로드, 총 {total_size / 1024 / 1024:.1f}MB"
    )
    return downloaded


# nslkdd 파일 ID 목록
nslkdd_files = {
    "1.npy": "1ikjxpHy0U8HuJFPuZepBcvfBphCk3WNw",
    "2.npy": "1jXAWtrZIbW_uAn762f25zxi8ebW9gNaY",
    "3.npy": "1WTMsyFLYPtcS6DOVPnxbDYhqNpUPuA1b",
    "4.npy": "1vYZo0NcewtLaL9MEuGqWCE_4U871K5kg",
    "5.npy": "138gRG5b7hrH5g87ea8E9JO9nZxzPXStZ",
    "6.npy": "1FQyBgWxvG-P8HoLIGf55ItYWWDyvVUqw",
    "8.npy": "1RbjtLsI5FKvnoV4swewv647vmWBuUJI3",
    "9.npy": "1YcHavS5XlxqcqtqKzlGDeCzB7m8KFJPa",
    "10.npy": "1Cl6vrIc79mBiY4WARVa6ZWNuraw9xfsI",
    "11.npy": "1FIc0_CxbS59De3hCEN1_TcPgoAiVxJDK",
    "12.npy": "1DoWCzqOIaTVWpQpb8SIsF2QFL6yp0YO_",
    "13.npy": "14lxlfqRs69XURdJnxJO0XY1J2bxfOgYc",
    "14.npy": "1lpFFoI-wYawQ76dwukcksScVyLqFN1iI",
    "15.npy": "1uGCvMv-G2NPuNyNJaWZ2AbrMzlwHqKP7",
    "16.npy": "1rYgt9NHPoKc58zrAwmW52T2a3vGguFuf",
    "17.npy": "16du7OyAPFwcASI7W8-nJwXxYUhtRLF_s",
    "18.npy": "17r64qC1OIy3XZ1SLCbUO2RMRb4GN5PWW",
    "19.npy": "1CSBZpjkt4tvTmlZGNxKWG_q3ymgA7TWF",
    "20.npy": "1QnM-7ltg_8lw_rgqdl7kblg-NbXOE68j",
    "21.npy": "1d-EkyHSbsV4Y9ivD-FGITaURmP0EIYUV",
    "22.npy": "13c27m_AX-oSvaXWMBkd6OI1gkpCyR-bY",
    "23.npy": "1wdTZq8TwquwrnR_sQXfV-76tOqOniYaF",
    "24.npy": "1qaHnGCk-rJ6ktR4TBCSfluVWUY8y7inN",
    "25.npy": "1k-i_Y8wEYpqThCxLqMOsjNMxG3lSnFnw",
    "26.npy": "1rP5Zdx7B1KlSesMQXwu0zG22xpl5nlAI",
    "27.npy": "1CYYPiKLORK029VNkj1yzB-ZBdLXkRSMa",
    "28.npy": "1Z6fBp7o006XmeEOkfNs5z3EvAiA7t9Ka",
    "29.npy": "1wHbx2UIdelBelIh8ikL0NQCvcm-LwtTz",
    "30.npy": "1dFCIb-8XMKXvmpppuGSzKvmkLhan_tWL",
    "31.npy": "1-GPh2KRQcVpby-XXURysd3wUvX4cGsHY",
    "32.npy": "1rBVYN6Xj1G7T1bcoeZEflwiesfsTyjeb",
    "33.npy": "10OS9Iqh7jbIsCZfuCNcl0YuJD6Q4CR-w",
    "34.npy": "1LWa1u_0h5wHgCGgqHmu9eluohBwMxvvO",
    "35.npy": "1SNaXR5N_28wFPdpA0P8QxkA530J2tAjh",
    "36.npy": "1UjalS96XJntPqeFC0RehYUP7r0iSa74o",
    "37.npy": "1g_cb37SlKNdwLc7YEo2L02_ctOlG4Tvm",
    "38.npy": "1Rvj0gBczDFSqTKddR6ByQ6su7Ol99uZj",
    "39.npy": "1QL22SfMTXYKZK8_W_7VNHFbCovVpuWTu",
    "40.npy": "1k1XTG3cRPmOMG-A-dE1rZIA3tIq8c6yF",
    "41.npy": "1768pe2dEnWfnsJW-P0y8-vlnb3PyVJDY",
    "42.npy": "1JPjtnjGyNpjHtW2ZGKh3Ot3LBNGfwwfa",
    "43.npy": "1bQdNq33ES7nfEfpWX5Ot3kU-Kb1fBoDm",
    "44.npy": "1E-Eto-5cf-3husqjtMCk65YG8yf5bbdp",
    "45.npy": "1pr7CWK0HW0DnQDn2GMbVwdeDgK9j4Opg",
}

# unswnb15 파일 ID 목록
unswnb15_files = {
    "1.npy": "1qwfrNLCuuYQctevaZZ7ZVhuhI92x94ZZ",
    "2.npy": "1P_sUJA6JivNSRNFlkqkrUrvfgMqHAACI",
    "3.npy": "1b8jxZzflxVc7Ac_Unq5OPfxUqroKDwMw",
    "4.npy": "1l3kMtBVEwiIvnr966KJjJUvXfrkcTPxj",
    "5.npy": "1h1HmdhkT27LFaUFEGBvBgosRK4L41BO4",
    "6.npy": "1TBzMPSDt7l21Vymc8yuolAq1bEPIvuuA",
    "7.npy": "1cMOAVsWwLiuudhmIQyyRov92W2ubh2-n",
    "8.npy": "1ufF1MDqZvGFkhndAl8j7GcBWOETKZFen",
    "9.npy": "158nuQnCs2LHTE1PkehsjM7zSCBW2Xpgr",
    "10.npy": "1iZbr0yU6TWfJkQ4Z6vUQODqWKvQrCfg_",
    "11.npy": "1OpUTIkksH4eH1GPTbl3KhIGi8MtApQqK",
    "12.npy": "1YGgqOBdc3iyh8NDo7py7ZXaisA8pcMW1",
    "13.npy": "1QpcwwiHCl_96wXjqxYNEOp-Hs0IiTOUQ",
    "14.npy": "1xoVVYlq0OqrfMle4tvuTauWoX9OiezBV",
    "15.npy": "1mwbjApaHp_k0fuQaSjfFvBleEYWKMb87",
    "16.npy": "1PvxfchkO5Ras-qW2TqLfjG71Jth-IVyp",
    "17.npy": "11Cg2eKjoU5J9zWXRX1PZfEG4NKJgKvW2",
    "18.npy": "19ATQhrPCBobR9ZRlThZkTONQWolLd1VO",
}

if __name__ == "__main__":
    print("=== ECBRS NIDS 데이터셋 다운로드 ===\n")

    print("1. ids17 다운로드 중...")
    download_dataset("ids17", ids17_files)

    print("\n2. ids18 다운로드 중...")
    download_dataset("ids18", ids18_files)

    print("\n3. nslkdd 다운로드 중...")
    download_dataset("nslkdd", nslkdd_files)

    print("\n4. unswnb15 다운로드 중...")
    download_dataset("unswnb15", unswnb15_files)

    print("\n=== 다운로드 완료 ===")
