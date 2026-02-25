/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        sipark: {
          bg: '#F4F6F8',
          primary: '#0F5E9C',
          text: '#1F2937',
          muted: '#6B7280',
          border: '#E5E7EB',
          danger: '#C62828',
          success: '#2E7D32',
          softBlue: '#E3F2FD',
          softDanger: '#FFEBEE',
          softSuccess: '#E8F5E9',
          tableAlt: '#FAFBFC',
        },
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.06)',
        subtle: '0 1px 2px rgba(0,0,0,0.04)',
      },
      borderRadius: {
        lg: '8px',
        md: '6px',
      },
      maxWidth: {
        layout: '1440px',
      },
    },
    screens: {
      sm: '640px',
      lg: '1024px',
    },
  },
  plugins: [],
}
