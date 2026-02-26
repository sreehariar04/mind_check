(() => {
    const navbar = document.getElementById('landingNavbar');
    const reveals = document.querySelectorAll('.mc-reveal');

    if (!navbar && reveals.length === 0) {
        return;
    }

    let previousScrollY = window.scrollY;

    const handleNavbarScroll = () => {
        if (!navbar) {
            return;
        }

        const currentScrollY = window.scrollY;
        const shouldHide = currentScrollY > previousScrollY && currentScrollY > 90;

        navbar.classList.toggle('nav-hidden', shouldHide);
        previousScrollY = currentScrollY;
    };

    if (navbar) {
        window.addEventListener('scroll', handleNavbarScroll, { passive: true });
    }

    if (reveals.length > 0) {
        const observer = new IntersectionObserver(
            (entries, observerRef) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) {
                        return;
                    }

                    entry.target.classList.add('is-visible');
                    observerRef.unobserve(entry.target);
                });
            },
            {
                threshold: 0.15,
                rootMargin: '0px 0px -10% 0px',
            }
        );

        reveals.forEach((element) => observer.observe(element));
    }
})();
