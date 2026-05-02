// 语音播报功能 - 基于 VoiceCraft API
const TTS_API_URL = 'https://tts.wangwangit.com/v1/audio/speech';
let currentAudio = null;
let isSpeaking = false;
let currentGender = localStorage.getItem('voiceGender') || 'female';

// 语音配置
const voiceConfig = {
    female: {
        voice: 'zh-CN-XiaoxiaoNeural',
        name: '晓晓',
        desc: '温柔女声'
    },
    male: {
        voice: 'zh-CN-YunxiNeural',
        name: '云希',
        desc: '清朗男声'
    }
};

// 初始化
function initTTS() {
    const savedVoice = localStorage.getItem('selectedVoice');
    if (savedVoice) {
        if (savedVoice.includes('Yunxi') || savedVoice.includes('Yunyang')) {
            currentGender = 'male';
        } else {
            currentGender = 'female';
        }
    }
    updateTTSUI();
}

// 切换语音
function switchVoice(gender) {
    currentGender = gender;
    const config = voiceConfig[gender];
    
    // 保存配置
    const ttsConfig = {
        voice: config.voice,
        speed: 1.0,
        pitch: '0',
        style: 'general',
        volume: '0'
    };
    
    localStorage.setItem('voiceGender', gender);
    localStorage.setItem('ttsConfig', JSON.stringify(ttsConfig));
    localStorage.setItem('selectedVoice', config.voice);
    
    updateTTSUI();
}

// 更新UI显示
function updateTTSUI() {
    const voice = voiceConfig[currentGender];
    var currentSpan = document.getElementById('tts-current-voice');
    if (currentSpan) {
        currentSpan.textContent = `当前: ${voice.name} (${voice.desc})`;
    }
}

// 切换语音（供按钮调用）
function toggleVoiceGender() {
    var gender = currentGender === 'female' ? 'male' : 'female';
    switchVoice(gender);
    
    var femaleBtn = document.getElementById('tts-female-btn');
    var maleBtn = document.getElementById('tts-male-btn');
    if (femaleBtn && maleBtn) {
        femaleBtn.classList.toggle('active', gender === 'female');
        maleBtn.classList.toggle('active', gender === 'male');
    }
}

// 朗读/停止
async function toggleSpeech() {
    var icon = document.getElementById('tts-icon');
    var text = document.getElementById('tts-text');
    
    if (isSpeaking) {
        // 停止播放
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = '';
            currentAudio = null;
        }
        icon.textContent = '🔊';
        text.textContent = '朗读全文';
        isSpeaking = false;
        return;
    }
    
    // 获取文章内容
    var content = document.getElementById('article-content');
    if (!content) {
        content = document.querySelector('.article-entry');
    }
    if (!content) {
        alert('未找到文章内容');
        return;
    }
    
    // 获取纯文本内容
    var articleText = content.innerText || content.textContent;
    
    // 清理文本（保留段落）
    articleText = articleText.replace(/\n{3,}/g, '\n\n').trim();
    
    if (!articleText) {
        alert('文章内容为空');
        return;
    }
    
    // 截断过长文本
    if (articleText.length > 5000) {
        articleText = articleText.substring(0, 5000) + '...（内容过长，已截断）';
    }
    
    icon.textContent = '⏳';
    text.textContent = '生成中...';
    
    try {
        // 获取配置
        var configStr = localStorage.getItem('ttsConfig');
        var config = configStr ? JSON.parse(configStr) : {
            voice: voiceConfig[currentGender].voice,
            speed: 1.0,
            pitch: '0',
            style: 'general',
            volume: '0'
        };
        
        // 调用 API
        var response = await fetch(TTS_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                input: articleText,
                ...config
            })
        });
        
        if (!response.ok) {
            throw new Error('API请求失败: ' + response.status);
        }
        
        var audioBlob = await response.blob();
        
        // 停止之前的音频
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = '';
        }
        
        // 创建音频
        var audioUrl = URL.createObjectURL(audioBlob);
        currentAudio = new Audio(audioUrl);
        
        currentAudio.onended = function() {
            URL.revokeObjectURL(audioUrl);
            currentAudio = null;
            icon.textContent = '🔊';
            text.textContent = '朗读全文';
            isSpeaking = false;
        };
        
        currentAudio.onerror = function(e) {
            URL.revokeObjectURL(audioUrl);
            currentAudio = null;
            icon.textContent = '🔊';
            text.textContent = '朗读全文';
            isSpeaking = false;
            console.error('音频播放错误:', e);
        };
        
        // 播放
        await currentAudio.play();
        icon.textContent = '⏹️';
        text.textContent = '停止朗读';
        isSpeaking = true;
        
    } catch (error) {
        console.error('TTS错误:', error);
        icon.textContent = '🔊';
        text.textContent = '朗读全文';
        isSpeaking = false;
        alert('语音生成失败: ' + error.message + '\n\n请检查网络连接后重试。');
    }
}

// 页面加载初始化
document.addEventListener('DOMContentLoaded', function() {
    initTTS();
});
