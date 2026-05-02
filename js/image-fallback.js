// 图片加载失败处理脚本
document.addEventListener('DOMContentLoaded', function() {
    var images = document.querySelectorAll('.article-entry img, .entry-content img');
    
    images.forEach(function(img) {
        // 图片加载失败处理
        img.onerror = function() {
            this.onerror = null;
            this.style.display = 'none';
            
            // 创建占位符
            var placeholder = document.createElement('div');
            placeholder.className = 'image-placeholder';
            placeholder.style.cssText = 
                'display: flex; ' +
                'flex-direction: column; ' +
                'align-items: center; ' +
                'justify-content: center; ' +
                'min-height: 150px; ' +
                'background: #2a0a0a; ' +
                'border: 1px dashed #ff4444; ' +
                'border-radius: 5px; ' +
                'margin: 15px auto; ' +
                'padding: 20px; ' +
                'max-width: 100%;';
            
            placeholder.innerHTML = 
                '<span style="font-size: 40px;">🖼️</span>' +
                '<span style="color: #ff4444; margin-top: 10px; font-size: 14px;">图片加载失败</span>' +
                '<span style="color: #888; margin-top: 5px; font-size: 12px;">' + 
                (this.src ? this.src.substring(0, 50) + '...' : '未知图片') + 
                '</span>';
            
            this.parentNode.insertBefore(placeholder, this);
        };
        
        // 图片加载成功处理
        img.onload = function() {
            this.style.opacity = '0';
            this.style.transition = 'opacity 0.3s';
            var self = this;
            requestAnimationFrame(function() {
                self.style.opacity = '1';
            });
        };
    });
});
