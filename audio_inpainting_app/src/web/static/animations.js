/**
 * DPAI Animation Controller
 * Handles scroll reveals, staggered animations, and micro-interactions
 * Respects prefers-reduced-motion
 */

(function () {
  'use strict';

  // Check for reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  
  if (prefersReducedMotion) {
    // Mark all reveal elements as visible immediately
    document.querySelectorAll('.reveal-on-scroll, .reveal-stagger > *').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
      el.style.transition = 'none';
    });
    return;
  }

  // =========================================================================
  // INTERSECTION OBSERVER FOR SCROLL REVEALS
  // =========================================================================
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        // Unobserve after revealing to save resources
        revealObserver.unobserve(entry.target);
      }
    });
  }, {
    root: null,
    rootMargin: '0px 0px -50px 0px', // Trigger 50px before element enters viewport
    threshold: 0.1
  });

  // Observe all elements with reveal classes
  function initScrollReveals() {
    document.querySelectorAll('.reveal-on-scroll').forEach(el => {
      revealObserver.observe(el);
    });
    
    document.querySelectorAll('.reveal-stagger').forEach(el => {
      revealObserver.observe(el);
    });
  }

  // =========================================================================
  // BUTTON RIPPLE EFFECT
  // =========================================================================
  function createRipple(event, button) {
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;

    const ripple = document.createElement('span');
    ripple.className = 'btn-ripple';
    ripple.style.cssText = `
      position: absolute;
      width: ${size}px;
      height: ${size}px;
      left: ${x}px;
      top: ${y}px;
      background: rgba(255, 255, 255, 0.25);
      border-radius: 50%;
      transform: scale(0);
      animation: ripple-anim 400ms var(--ease-out-expo);
      pointer-events: none;
      z-index: 1;
    `;

    // Ensure button has relative positioning
    const computedStyle = window.getComputedStyle(button);
    if (computedStyle.position === 'static') {
      button.style.position = 'relative';
    }
    button.style.overflow = 'hidden';

    button.appendChild(ripple);

    ripple.addEventListener('animationend', () => {
      ripple.remove();
    });
  }

  // Add ripple to primary buttons
  function initButtonRipples() {
    const rippleSelectors = [
      '.btn-primary-recon',
      '.btn-danger-rack',
      '.btn-rack-download',
      '.btn-rack-tool',
      '.btn-rack-mini',
      '.btn-ab-toggle',
      '.tab-toggle',
      '.preset-choice'
    ];

    rippleSelectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(btn => {
        if (btn.disabled) return;
        btn.addEventListener('click', (e) => createRipple(e, btn));
      });
    });
  }

  // Add ripple keyframes if not present
  function addRippleStyles() {
    if (document.getElementById('ripple-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'ripple-styles';
    style.textContent = `
      @keyframes ripple-anim {
        to {
          transform: scale(2.5);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);
  }

  // =========================================================================
  // GAP CHIP ENTRANCE STAGGER
  // =========================================================================
  function animateGapChips() {
    const chips = document.querySelectorAll('.gap-chip');
    chips.forEach((chip, index) => {
      chip.style.animationDelay = `${index * 60}ms`;
    });
  }

  // =========================================================================
  // METRIC CARD ENTRANCE STAGGER
  // =========================================================================
  function animateMetricCards() {
    const cards = document.querySelectorAll('.metric-block');
    cards.forEach((card, index) => {
      card.style.animationDelay = `${index * 80}ms`;
    });
  }

  // =========================================================================
  // TOOLTIP INITIALIZATION
  // =========================================================================
  function initTooltips() {
    // Add data-tooltip attributes to elements with title attributes
    document.querySelectorAll('[title]').forEach(el => {
      const title = el.getAttribute('title');
      if (title) {
        el.setAttribute('data-tooltip', title);
        // Don't remove title entirely - keep for accessibility
        // el.removeAttribute('title'); // Commented out for accessibility
      }
    });
  }

  // =========================================================================
  // LIVE SPEC TRANSITION
  // =========================================================================
  function initLiveSpecTransition() {
    const liveSpecImg = document.getElementById('live-spec-img');
    const specPlaceholder = document.getElementById('spec-placeholder-text');
    
    if (!liveSpecImg || !specPlaceholder) return;

    liveSpecImg.addEventListener('load', () => {
      liveSpecImg.classList.add('active');
      specPlaceholder.classList.add('hidden');
    });
  }

  // =========================================================================
  // PROGRESS BAR ENHANCED ANIMATION
  // =========================================================================
  function initProgressBar() {
    const progressBar = document.getElementById('progress-bar-fill');
    if (!progressBar) return;

    // Observe progress bar for smooth width animation
    const progressObserver = new MutationObserver(() => {
      const width = progressBar.style.width;
      if (width) {
        progressBar.style.transition = 'width 200ms var(--ease-out-expo)';
      }
    });
    
    progressObserver.observe(progressBar, { attributes: true, attributeFilter: ['style'] });
  }

  // =========================================================================
  // KEYBOARD FOCUS VISUAL ENHANCEMENT
  // =========================================================================
  function initFocusStyles() {
    // Add focus-visible polyfill behavior for older browsers
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        document.body.classList.add('keyboard-nav');
      }
    });

    document.addEventListener('mousedown', () => {
      document.body.classList.remove('keyboard-nav');
    });
  }

  // =========================================================================
  // INITIALIZE ALL ANIMATIONS
  // =========================================================================
  function init() {
    addRippleStyles();
    initScrollReveals();
    initButtonRipples();
    initTooltips();
    initLiveSpecTransition();
    initProgressBar();
    initFocusStyles();
    
    // Animate existing elements on load
    animateGapChips();
    animateMetricCards();
    
    // Re-animate when new gap chips are added
    const gapsList = document.getElementById('gaps-list');
    if (gapsList) {
      const listObserver = new MutationObserver(() => {
        animateGapChips();
      });
      listObserver.observe(gapsList, { childList: true, subtree: true });
    }
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export for manual triggering if needed
  window.DPAIAnimations = {
    animateGapChips,
    animateMetricCards,
    initScrollReveals,
    initButtonRipples
  };
})();